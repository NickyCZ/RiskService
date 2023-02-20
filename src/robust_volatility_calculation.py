import boto3
import pandas as pd
import numpy as np
from datetime import datetime
from boto3.dynamodb.conditions import Key
from aws_lambda_powertools import Logger
from aws_lambda_powertools.event_handler import APIGatewayRestResolver
from aws_lambda_powertools.logging import correlation_paths
from aws_lambda_powertools.utilities.typing import LambdaContext

logger = Logger()
app = APIGatewayRestResolver()
dynamodb = boto3.resource("dynamodb")

@app.post("/robust_volatility_calculation")
def robust_volatility_calculation():
    data = app.current_event.json_body
    instrument = data['instrument']
    start_time = int(data["start_time"])

    raw_prices = retrieve_prices_from_dynamodb(instrument, start_time)
    prices = aggregate_to_day_based_prices(raw_prices)

    # Standard deviation will be nan for first 10 non nan values
    volatility = simple_ewvol_calc(prices)
    cutted_vol = apply_min_vol(volatility)

    floored_volatility = apply_vol_floor(cutted_vol)
    # Return a confirmation message and the input parameters as the result of the function
    return {
        'rule': 'Robust Volatility Series',
        'instrument': instrument,
    }

def retrieve_prices_from_dynamodb(instrument: str, start_time: int) -> dict: 
    table_name = "multiple_prices"
    end_time = int(datetime.now().timestamp())
    try:
        table = dynamodb.Table(table_name)
        items = []
        response = table.query(
            KeyConditionExpression=Key('Instrument').eq(instrument) & Key('UnixDateTime').between(start_time, end_time),
            ProjectionExpression='UnixDateTime,Price'
        )
        items += response['Items']
        while 'LastEvaluatedKey' in response:
            response = table.query(
                KeyConditionExpression=Key('Instrument').eq(instrument) & Key('UnixDateTime').between(start_time, end_time),
                ProjectionExpression='UnixDateTime,Price',
                ExclusiveStartKey=response['LastEvaluatedKey']
            )
            items += response['Items']
        return items

    except Exception as e:
        # log the error message
        logger.error("Error occurred while retrieving prices from DynamoDB", e)
        raise ValueError("Error occurred while retrieving prices from DynamoDB")

def aggregate_to_day_based_prices(raw_prices: dict) -> pd.Series:
    df = pd.DataFrame.from_dict(raw_prices)
    series = df.set_index('UnixDateTime')
    series.index = pd.to_datetime(pd.to_numeric(series.index), unit='s')
    series['Price'] = pd.to_numeric(series['Price'])
    daily_summary = series.resample('D').mean()
    return daily_summary

def simple_ewvol_calc(daily_returns: pd.Series) -> pd.Series:

    days = 35
    min_periods = 10
    # Standard deviation will be nan for first 10 non nan values
    vol = daily_returns.ewm(adjust=True, span=days, min_periods=min_periods).std()
    return vol

def apply_vol_floor(vol: pd.Series, floor_min_quant: float = 0.05, floor_min_periods: int = 100, floor_days: int = 500,) -> pd.Series:
    # Find the rolling 5% quantile point to set as a minimum
    vol_min = vol.rolling(min_periods=floor_min_periods, window=floor_days).quantile(
        quantile=floor_min_quant
    )

    # set this to zero for the first value then propagate forward, ensures
    # we always have a value
    vol_min.iloc[0] = 0.0
    vol_min.ffill(inplace=True)

    # apply the vol floor
    vol_floored = np.maximum(vol, vol_min)

    return vol_floored

def apply_min_vol(vol: pd.Series) -> pd.Series:
    vol_abs_min= 0.0000000001
    vol[vol < vol_abs_min] = vol_abs_min
    return vol

@logger.inject_lambda_context(correlation_id_path=correlation_paths.API_GATEWAY_REST)
def lambda_handler(event: dict, context: LambdaContext) -> dict:
    return app.resolve(event, context)
