
## conversion factors into meters
m  = 1.0
cm = 0.01
mm = 0.001

## map from names to values
_name_dict = {
    "m":  m,
    "cm": cm,
    "mm": mm
}

## map describing alternative names for each unit
_alternative_name_dict = {
    "m":  ["m", "meter", "metre", "M", "Meter", "Metre"],
    "cm": ["cm", "centimeter", "centimetre", "CM", "Centimeter", "Centimetre"],
    "mm": ["mm", "millimeter", "millimetre", "MM", "Millimeter", "Millimetre"] 
}

def from_string(string: str) -> float:
    """Convert string describing a measurement (number with optional unit) into a numerical value (in meters)

    e.g. "5 mm" will be converted to 0.005

    if no unit is provided, value is assumed to be in meters so 

    "10" will be converted to 10.0

    :param string: String describing the value
    :type string: str
    :return: the value described by the string (in meters)
    :rtype: float
    :raises: ValueError if the string is not in the right format or the unit name is not recognised
    """

    split = string.split(" ")
    if len(split) > 2:
        print(f"ERROR: invalid measurement string: {string} - too many values provided - should be either [value] [unit] or [value]")
        raise ValueError("Bad units")
    if len(split) == 0:
        print(f"ERROR: invalid measurement string: {string} - not enough values provided - should be either [value] [unit] or [value]")
        raise ValueError("Bad units")
    
    try:
        value = float(split[0])

    except:
        print(f"ERROR: invalid measurement string: {string} - value is not numeric - should be either [value(float)] [unit(string)] or [value(float)]")
        raise ValueError("Bad unit value")

    ## if only value was provided, assume meters
    if len(split) == 1:
        return value
    
    unit = None
    for unit_name, alt_names in zip(_alternative_name_dict.keys(), _alternative_name_dict.values()):
        if split[1] in alt_names:
            unit = unit_name
            break

    if unit is None: 
        print(f'ERROR: Unit name {split[1]} was not recognised (in string {string})')
        print(f'Only the following units are recognised:')
        for alt_names in _alternative_name_dict:
            print(f'  - {alt_names}')

        raise ValueError("Bad unit value")
    
    return _name_dict[unit] * value
