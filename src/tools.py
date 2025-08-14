from langchain_core.tools import tool
import math


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