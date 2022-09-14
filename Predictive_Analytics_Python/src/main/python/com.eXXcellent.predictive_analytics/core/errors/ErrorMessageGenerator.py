from enum import Enum


class ErrorTypes(Enum):
    InvalidAction = 0,
    NoFileFound = 1,
    ResourceNotFound = 2,
    MethodNotFound = 3,
    CaseNotImplemented = 4,
    MissingParameter = 5


def generate_error_message(error_type, *args):
    if error_type == ErrorTypes.InvalidAction:
        return f"{args[0]} is not a valid action. Valid actions are {args[1]}"
    elif error_type == ErrorTypes.NoFileFound:
        return f"No valid file in form-data body with key {args[0]} found. Valid files are {args[1]}"
    elif error_type == ErrorTypes.ResourceNotFound:
        return f"The Resource {args[0]} with id {args[1]} was not found."
    elif error_type == ErrorTypes.MethodNotFound:
        return f"The requested action {args[0]} is not allowed."
    elif error_type == ErrorTypes.CaseNotImplemented:
        return f"The case {args[0]} has not been implemented. {args[1]}"
    elif error_type == ErrorTypes.MissingParameter:
        return f"The parameter {args[0]} is missing or empty. Please add it to the request body in the following " \
               f"format: {args[1]} "
    else:
        return f"The error message for {error_type} does not exist."
