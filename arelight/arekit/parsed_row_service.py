from arekit.common.data import const
from arekit.common.data.rows_parser import ParsedSampleRow


class ParsedSampleRowExtraService(object):
    """ This is a specific extension for ParsedRow of AREkit library.
        It provides other values that might be calculated and based on
        the predefined constant fields of the AREkit data source.
    """

    __service = {
        "SourceValue": lambda parsed_row: ParsedSampleRowExtraService.calc_obj_value(
            obj_id=parsed_row[const.S_IND],
            obj_ids=parsed_row[const.ENTITIES],
            obj_values=parsed_row[const.ENTITY_VALUES]),
        "TargetValue": lambda parsed_row: ParsedSampleRowExtraService.calc_obj_value(
            obj_id=parsed_row[const.T_IND],
            obj_ids=parsed_row[const.ENTITIES],
            obj_values=parsed_row[const.ENTITY_VALUES]),
        "SourceType": lambda parsed_row: ParsedSampleRowExtraService.calc_obj_value(
            obj_id=parsed_row[const.S_IND],
            obj_ids=parsed_row[const.ENTITIES],
            obj_values=parsed_row[const.ENTITY_TYPES]),
        "TargetType": lambda parsed_row: ParsedSampleRowExtraService.calc_obj_value(
            obj_id=parsed_row[const.T_IND],
            obj_ids=parsed_row[const.ENTITIES],
            obj_values=parsed_row[const.ENTITY_TYPES]),
    }

    @staticmethod
    def calc_obj_value(obj_id, obj_values, obj_ids):
        assert(isinstance(obj_values, list))
        assert(isinstance(obj_ids, list))
        assert(len(obj_values) == len(obj_ids))

        ind = obj_ids.index(obj_id)
        if ind < 0:
            return None

        return obj_values[ind]

    @staticmethod
    def calc(service_name, parsed_row):
        assert (isinstance(parsed_row, ParsedSampleRow))
        calc_func = ParsedSampleRowExtraService.__service[service_name]
        return calc_func(parsed_row)
