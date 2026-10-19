import enum

import lps_ml.datasets.synthetic as ml_syn
import lps_ml.core.selection as ml_sel
import lps_ml.datasets as ml_db

import acoustic_simulator.default as as_default

class SyntheticExperiment(enum.Enum):
    DEPTH = enum.auto()
    BOTTOM = enum.auto()
    DYNAMIC = enum.auto()
    SEASON = enum.auto()
    VOLUME = enum.auto()

    def as_str(self) -> str:
        return self.name.lower()

    def get_filters(self) -> dict[str, list[ml_sel.Filter]]:

        if self == SyntheticExperiment.DEPTH:
            return {
                "shallow": [ml_sel.ConstraintFilter(
                    constraints={
                        "header": "SHALLOW_WATER",
                        "value": ["yes"]
                    },
                    remove_elements_in = False
                )],
                "deep": [ml_sel.ConstraintFilter(
                    constraints={
                        "header": "SHALLOW_WATER",
                        "value": ["no"]
                    },
                    remove_elements_in = False
                )]
            }

        if self == SyntheticExperiment.BOTTOM:

            shallow_filter = ml_sel.ConstraintFilter(
                constraints={
                    "header": "SHALLOW_WATER",
                    "value": ["yes"]
                },
                remove_elements_in=False
            )

            return {
                "basalt": [
                    shallow_filter,
                    ml_sel.ConstraintFilter(
                    constraints={
                        "header": "SEABED",
                        "value": ["Basalt"]
                    },
                    remove_elements_in = False
                )],
                "clay": [
                    shallow_filter,
                    ml_sel.ConstraintFilter(
                    constraints={
                        "header": "SEABED",
                        "value": ["Clay"]
                    },
                    remove_elements_in = False
                )],
                "gravel": [
                    shallow_filter,
                    ml_sel.ConstraintFilter(
                    constraints={
                        "header": "SEABED",
                        "value": ["Gravel"]
                    },
                    remove_elements_in = False
                )],
                "sand": [
                    shallow_filter,
                    ml_sel.ConstraintFilter(
                    constraints={
                        "header": "SEABED",
                        "value": ["Sand"]
                    },
                    remove_elements_in = False
                )],
                "silt": [
                    shallow_filter,
                    ml_sel.ConstraintFilter(
                    constraints={
                        "header": "SEABED",
                        "value": ["Silt"]
                    },
                    remove_elements_in = False
                )],
            }

        if self == SyntheticExperiment.DYNAMIC:
            return {
                "fixed distance": [ml_sel.ConstraintFilter(
                    constraints={
                        "header": "DYNAMIC_TYPE",
                        "value": ["fixed_distance"]
                    },
                    remove_elements_in = False
                )],
                "cpa in": [ml_sel.ConstraintFilter(
                    constraints={
                        "header": "DYNAMIC_TYPE",
                        "value": ["cpa_in"]
                    },
                    remove_elements_in = False
                )],
                "cpa out": [ml_sel.ConstraintFilter(
                    constraints={
                        "header": "DYNAMIC_TYPE",
                        "value": ["cpa_out"]
                    },
                    remove_elements_in = False
                )],
            }

        if self == SyntheticExperiment.SEASON:
            return {
                "summer": [ml_sel.ConstraintFilter(
                    constraints={
                        "header": "Season",
                        "value": ["Summer"]
                    },
                    remove_elements_in = False
                )],
                "autumn": [ml_sel.ConstraintFilter(
                    constraints={
                        "header": "Season",
                        "value": ["Autumn"]
                    },
                    remove_elements_in = False
                )],
                "winter": [ml_sel.ConstraintFilter(
                    constraints={
                        "header": "Season",
                        "value": ["Winter"]
                    },
                    remove_elements_in = False
                )],
                "spring": [ml_sel.ConstraintFilter(
                    constraints={
                        "header": "Season",
                        "value": ["Spring"]
                    },
                    remove_elements_in = False
                )],
            }

        if self == SyntheticExperiment.VOLUME:
            return {
                "reference": [ml_sel.ConstraintFilter(
                    constraints={
                        "header": "SCENARIO_CATALOG_ID",
                        "value": [14, 3]
                    },
                    remove_elements_in = False
                )],
                "non reference": [ml_sel.ConstraintFilter(
                    constraints={
                        "header": "SCENARIO_CATALOG_ID",
                        "value": [14, 3]
                    },
                    remove_elements_in = True
                )],
            }

        raise NotImplementedError(f"get_filters not implemented for {self}.")

    def get_dynamic_selection(self) -> ml_syn.DynamicSelection:
        return ml_syn.DynamicSelection.ALL

    def get_channel_selection(self) -> ml_syn.ChannelSelection:
        if self in [SyntheticExperiment.BOTTOM, SyntheticExperiment.SEASON]:
            return ml_syn.ChannelSelection.NON_REFERENCE_ONLY
        return ml_syn.ChannelSelection.ALL

    def build_datamodules(self) -> dict[str, ml_db.Iemanja]:

        datamodules = {}


        for name, filters in self.get_filters().items():

            try:

                file_processor = as_default.get_file_processor()
                cv = as_default.get_cv()

                selection = ml_sel.Selector(
                    target=ml_sel.ColumnTarget(
                        column="CLASS",
                        map_values=True,
                        mapping = {
                            'Service': 0,
                            'Cargo': 1
                        }
                    ),
                    filters=filters,
                )

                datamodules[name] = ml_db.Iemanja(
                    file_processor=file_processor,
                    selection=selection,
                    cv=cv,
                    dynamic_selection=self.get_dynamic_selection(),
                    channel_selection=self.get_channel_selection(),
                )

            except UnboundLocalError:
                continue


        return datamodules
