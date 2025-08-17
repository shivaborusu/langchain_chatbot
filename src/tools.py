from langchain_core.tools import tool
import math
from langchain_community.tools import BraveSearch
import os
from dotenv import load_dotenv
load_dotenv()


def web_search():
    return BraveSearch.from_api_key(api_key=os.environ.get('BRAVE_API_KEY'),
                                 search_kwargs={"count": 1})


@tool
def multiply(x: float, y: float) -> float:
    """ Mulitply 'x' times 'y' """
    return x*y


@tool
def add(x: float, y: float) -> float:
    """ Add 'x' and 'y' """
    return x+y


@tool
def exponentiate(x: float, y: float) -> float:
    """ Raise y to the power of y,
    also called exponentiation """
    return math.pow(x,y)