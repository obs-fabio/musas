import enum
import lps_ml.core.datamodule as ml_core
import lps_ml.datasets as ml_db
import lps_ml.datasets.iara as ml_iara

import acoustic_simulator.default as as_default

class DomainExperiment:

    def __init__(self):
        self.file_processor = as_default.get_file_processor()
        self.cv = as_default.get_cv()

    def build_iara(self):
        return ml_db.IARA(
            file_processor=self.file_processor,
            data_collection=ml_iara.DataCollection.OS,
            cv=self.cv,
            selection=ml_iara.CargoShipClassifier.GENERAL.as_selector(),
        )

    def build_iemanja(self):
        return ml_db.Iemanja(
            file_processor=self.file_processor,
            cv=self.cv,
        )

    def build_domain_datamodules(self) -> dict[str, ml_core.BaseDataModule]:
        return {
            "iara": self.build_iara(),
            "iemanja": self.build_iemanja(),
        }

    def build_combined_datamodule(self) -> ml_core.CombinedDataModule:
        combined = ml_core.CombinedDataModule(self.build_domain_datamodules())
        return combined
