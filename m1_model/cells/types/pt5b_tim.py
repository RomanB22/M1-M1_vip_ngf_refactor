# m1_model/cells/types/pt5b_tim.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
import json

from m1_model.utils.param_patch import atomic_write_json

from m1_model.cells.base import ImportSpec, CellProvider
from m1_model.utils.csv_helpers import csv_to_dict
from m1_model.utils.param_patch import apply_na_paramfile_to_rule


class PT5BFullTimFromPy(CellProvider):
    """
    Additional PT5B model (“Tim” variant) under the canonical label PT5B_full.

    Pipeline:
      1) Load mutant params CSV selected by cfg.variant (or 'WT') and write the JSON
         file that the Na12 model expects.
      2) importCellParams from Na12HMMModel_TF.py (class Na12Model_TF), soma at origin.
      3) Post-processing:
         - Rename soma_0 -> soma
         - Set spikeGenLoc on axon_0
         - Inject pt3d points for axon_0 and axon_1 (keeps SectionLists robust)
         - Reset/compute secLists: perisom, below_soma, alldend, apicdend, spiny
         - Heterozygous / blockNa toggles (na12/na12mut gbar)
         - Ih scaling via cfg.ihGbar (+ cfg.ihGbarBasal on dend*), skip axon_*
         - Reduce apical (dendritic) Na via cfg.dendNa
         - Add weight normalization; optional save to JSON
    """

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.ctx = None  # set in import_spec

    # -------------------------- Post-processing hook --------------------------

    def _post(self, netParams):
        cfg = self.ctx.cfg
        label = "PT5B_full"
        rule = netParams.cellParams[label]

        # 1) rename soma_0 -> soma
        netParams.renameCellParamsSec(label=label, oldSec="soma_0", newSec="soma")

        # 2) spike generation site on axon_0
        rule.setdefault("secs", {}).setdefault("axon_0", {}).setdefault("spikeGenLoc", 0.5)

        # 3) add pt3d to axon sections so SecList ops remain stable
        ax0_geom = rule["secs"].setdefault("axon_0", {}).setdefault("geom", {})
        ax1_geom = rule["secs"].setdefault("axon_1", {}).setdefault("geom", {})
        ax0_geom["pt3d"] = [
            [-25.435224533081055, 34.14994812011719, 0, 1.6440753936767578],
            [-25.065839767456055, 34.10675811767578, 0, 1.6440753936767578],
            [-24.327072143554688, 34.02037811279297, 0, 1.6440753936767578],
            [-23.588302612304688, 33.933998107910156, 0, 1.6440753936767578],
            [-22.849533081054688, 33.847618103027344, 0, 1.6440753936767578],
            [-22.11076545715332, 33.7612419128418, 0, 1.6440753936767578],
            [-21.37199592590332, 33.674861907958984, 0, 1.6440753936767578],
            [-20.63322639465332, 33.58848190307617, 0, 1.6440753936767578],
            [-19.894458770751953, 33.50210189819336, 0, 1.6440753936767578],
            [-19.155689239501953, 33.41572189331055, 0, 1.6440753936767578],
            [-18.416919708251953, 33.329341888427734, 0, 1.6440753936767578],
            [-17.678152084350586, 33.24296188354492, 0, 1.6440753936767578],
            [-16.939382553100586, 33.15658187866211, 0, 1.6440753936767578],
            [-16.200613021850586, 33.0702018737793, 0, 1.6440753936767578],
            [-15.461844444274902, 32.98382568359375, 0, 1.6440753936767578],
            [-14.723075866699219, 32.89744567871094, 0, 1.6440753936767578],
            [-13.984307289123535, 32.811065673828125, 0, 1.6440753936767578],
            [-13.245537757873535, 32.72468566894531, 0, 1.6440753936767578],
            [-12.506769180297852, 32.6383056640625, 0, 1.6440753936767578],
            [-11.768000602722168, 32.55192565917969, 0, 1.6440753936767578],
            [-11.029231071472168, 32.465545654296875, 0, 1.6440753936767578],
            [-10.290462493896484, 32.37916564941406, 0, 1.6440753936767578],
            [-9.5516939163208, 32.29278564453125, 0, 1.6440753936767578],
            [-8.8129243850708, 32.2064094543457, 0, 1.6440753936767578],
            [-8.074155807495117, 32.12002944946289, 0, 1.6440753936767578],
            [-7.335386753082275, 32.03364944458008, 0, 1.6440753936767578],
            [-6.596618175506592, 31.947269439697266, 0, 1.6440753936767578],
            [-5.85784912109375, 31.860889434814453, 0, 1.6440753936767578],
            [-5.119080066680908, 31.77450942993164, 0, 1.6440753936767578],
            [-4.380311489105225, 31.68813133239746, 0, 1.6440753936767578],
            [-3.641542434692383, 31.60175132751465, 0, 1.6440753936767578],
            [-2.90277361869812, 31.515371322631836, 0, 1.6440753936767578],
            [-2.1640045642852783, 31.428991317749023, 0, 1.6440753936767578],
            [-1.4252357482910156, 31.342613220214844, 0, 1.6440753936767578],
            [-0.6864668726921082, 31.25623321533203, 0, 1.6440753936767578],
            [0.05230199918150902, 31.16985321044922, 0, 1.6440753936767578],
            [0.7910708785057068, 31.083473205566406, 0, 1.6440753936767578],
            [1.5298397541046143, 30.997093200683594, 0, 1.6440753936767578],
            [2.268608570098877, 30.910715103149414, 0, 1.6440753936767578],
            [3.0073776245117188, 30.8243350982666, 0, 1.6440753936767578],
            [3.7461464405059814, 30.73795509338379, 0, 1.6440753936767578],
            [4.484915256500244, 30.651575088500977, 0, 1.6440753936767578],
            [5.223684310913086, 30.565196990966797, 0, 1.6440753936767578],
            [5.9624528884887695, 30.478816986083984, 0, 1.6440753936767578],
            [6.701221942901611, 30.392436981201172, 0, 1.6440753936767578],
            [7.439990997314453, 30.30605697631836, 0, 1.6440753936767578],
            [8.178759574890137, 30.21967887878418, 0, 1.6440753936767578],
            [8.91752815246582, 30.133298873901367, 0, 1.6440753936767578],
            [9.65629768371582, 30.046918869018555, 0, 1.6440753936767578],
            [10.395066261291504, 29.960538864135742, 0, 1.6440753936767578],
            [11.133834838867188, 29.87415885925293, 0, 1.6440753936767578],
            [11.872604370117188, 29.78778076171875, 0, 1.6440753936767578],
            [12.611372947692871, 29.701400756835938, 0, 1.6440753936767578],
            [13.350141525268555, 29.615020751953125, 0, 1.6440753936767578],
            [14.088911056518555, 29.528640747070312, 0, 1.6440753936767578],
            [14.827679634094238, 29.442262649536133, 0, 1.6440753936767578],
            [15.566448211669922, 29.35588264465332, 0, 1.6440753936767578],
            [16.305217742919922, 29.269502639770508, 0, 1.6440753936767578],
            [17.043987274169922, 29.183122634887695, 0, 1.6440753936767578],
            [17.78275489807129, 29.096744537353516, 0, 1.6440753936767578],
            [18.52152442932129, 29.010364532470703, 0, 1.6440753936767578],
            [19.260292053222656, 28.92398452758789, 0, 1.6440753936767578],
            [19.999061584472656, 28.837604522705078, 0, 1.6440753936767578],
            [20.737831115722656, 28.751224517822266, 0, 1.6440753936767578],
            [21.476598739624023, 28.664846420288086, 0, 1.6440753936767578],
            [22.215368270874023, 28.578466415405273, 0, 1.6440753936767578],
            [22.954137802124023, 28.49208641052246, 0, 1.6440753936767578],
            [23.69290542602539, 28.40570640563965, 0, 1.6440753936767578],
            [24.43167495727539, 28.31932830810547, 0, 1.6440753936767578],
            [25.17044448852539, 28.232948303222656, 0, 1.6440753936767578],
            [25.909212112426758, 28.146568298339844, 0, 1.6440753936767578],
            [26.647981643676758, 28.06018829345703, 0, 1.6440753936767578],
            [27.386751174926758, 27.97381019592285, 0, 1.6440753936767578],
            [28.125518798828125, 27.88743019104004, 0, 1.6440753936767578],
            [28.864288330078125, 27.801050186157227, 0, 1.6440753936767578],
            [29.603057861328125, 27.714670181274414, 0, 1.6440753936767578],
            [30.341825485229492, 27.6282901763916, 0, 1.6440753936767578],
            [31.080595016479492, 27.541912078857422, 0, 1.6440753936767578],
            [31.819364547729492, 27.45553207397461, 0, 1.6440753936767578],
            [32.55813217163086, 27.369152069091797, 0, 1.6440753936767578],
            [33.29690170288086, 27.282772064208984, 0, 1.6440753936767578],
            [34.03567123413086, 27.196393966674805, 0, 1.6440753936767578],
            [34.77444076538086, 27.110013961791992, 0, 1.6440753936767578],
            [35.51321029663086, 27.02363395690918, 0, 1.6440753936767578],
            [36.251976013183594, 26.937253952026367, 0, 1.6440753936767578],
            [36.990745544433594, 26.850875854492188, 0, 1.6440753936767578],
            [37.729515075683594, 26.764495849609375, 0, 1.6440753936767578],
            [38.468284606933594, 26.678115844726562, 0, 1.6440753936767578],
            [39.207054138183594, 26.59173583984375, 0, 1.6440753936767578],
            [39.945823669433594, 26.505355834960938, 0, 1.6440753936767578],
            [40.68458938598633, 26.418977737426758, 0, 1.6440753936767578],
            [41.42335891723633, 26.332597732543945, 0, 1.6440753936767578],
            [42.16212844848633, 26.246217727661133, 0, 1.6440753936767578],
            [42.90089797973633, 26.15983772277832, 0, 1.6440753936767578],
            [43.63966751098633, 26.07345962524414, 0, 1.6440753936767578],
            [44.37843322753906, 25.987079620361328, 0, 1.6440753936767578],
            [45.11720275878906, 25.900699615478516, 0, 1.6440753936767578],
            [45.85597229003906, 25.814319610595703, 0, 1.6440753936767578],
            [46.59474182128906, 25.727941513061523, 0, 1.6440753936767578],
            [47.33351135253906, 25.64156150817871, 0, 1.6440753936767578],
            [48.07228088378906, 25.5551815032959, 0, 1.6440753936767578],
            [48.8110466003418, 25.468801498413086, 0, 1.6440753936767578],
            [49.5498161315918, 25.382421493530273, 0, 1.6440753936767578],
            [50.2885856628418, 25.296043395996094, 0, 1.6440753936767578],
            [51.0273551940918, 25.20966339111328, 0, 1.6440753936767578],
            [51.7661247253418, 25.12328338623047, 0, 1.6440753936767578],
            [52.5048942565918, 25.036903381347656, 0, 1.6440753936767578],
            [53.24365997314453, 24.950525283813477, 0, 1.6440753936767578],
            [53.98242950439453, 24.864145278930664, 0, 1.6440753936767578],
            [54.72119903564453, 24.77776527404785, 0, 1.6440753936767578],
            [55.45996856689453, 24.69138526916504, 0, 1.6440753936767578],
            [56.19873809814453, 24.60500717163086, 0, 1.6440753936767578],
            [56.93750762939453, 24.518627166748047, 0, 1.6440753936767578],
            [57.676273345947266, 24.432247161865234, 0, 1.6440753936767578],
            [58.415042877197266, 24.345867156982422, 0, 1.6440753936767578],
            [59.153812408447266, 24.25948715209961, 0, 1.6440753936767578],
            [59.892581939697266, 24.17310905456543, 0, 1.6440753936767578],
            [60.631351470947266, 24.086729049682617, 0, 1.6440753936767578],
            [61.370121002197266, 24.000349044799805, 0, 1.6440753936767578],
            [62.10888671875, 23.913969039916992, 0, 1.6440753936767578],
            [62.84765625, 23.827590942382812, 0, 1.6440753936767578],
            [63.58642578125, 23.7412109375, 0, 1.6440753936767578],
            [63.955810546875, 23.698020935058594, 0, 1.6440753936767578],
        ]
        ax1_geom["pt3d"] = [
            [63.955810546875, 23.698020935058594, 0, 1.6440753936767578],
            [65.31021881103516, 23.539657592773438, 0, 1.6440753936767578],
            [68.01904296875, 23.222932815551758, 0, 1.6440753936767578],
            [70.72785949707031, 22.906208038330078, 0, 1.6440753936767578],
            [73.43667602539062, 22.5894832611084, 0, 1.6440753936767578],
            [76.14550018310547, 22.27275848388672, 0, 1.6440753936767578],
            [78.85431671142578, 21.956031799316406, 0, 1.6440753936767578],
            [81.5631332397461, 21.639307022094727, 0, 1.6440753936767578],
            [84.27195739746094, 21.322582244873047, 0, 1.6440753936767578],
            [86.98077392578125, 21.005857467651367, 0, 1.6440753936767578],
            [89.68959045410156, 20.689132690429688, 0, 1.6440753936767578],
            [92.3984146118164, 20.372407913208008, 0, 1.6440753936767578],
            [93.75282287597656, 20.21404457092285, 0, 1.6440753936767578],
        ]

        # 4) conds (explicit)
        rule["conds"] = {"cellModel": "HH_full", "cellType": "PT"}

        # 5) (re)build useful SectionLists
        rule["secLists"] = {}
        nonSpiny = ["apic_0", "apic_1"]

        netParams.addCellParamsSecList(label=label, secListName="perisom", somaDist=[0, 50])
        netParams.addCellParamsSecList(label=label, secListName="below_soma", somaDistY=[-600, 0])

        names = list(rule["secs"].keys())
        alldend = [s for s in names if ("dend" in s or "apic" in s)]
        apicdend = [s for s in names if "apic" in s]
        spiny = [s for s in alldend if s not in nonSpiny]

        rule["secLists"]["alldend"] = alldend
        rule["secLists"]["apicdend"] = apicdend
        rule["secLists"]["spiny"] = spiny
        perisom = rule["secLists"].get("perisom", [])
        rule["secLists"]["perisom"] = [s for s in perisom if s not in nonSpiny]

        # 7) Ih scaling (skip axon_*)
        if hasattr(cfg, "ihGbar"):
            for secName, sec in rule["secs"].items():
                if secName in ("axon_0", "axon_1"):
                    continue
                Ih = sec.get("mechs", {}).get("Ih")
                if Ih and "gIhbar" in Ih:
                    g = Ih["gIhbar"]
                    scaled = [v * cfg.ihGbar for v in g] if isinstance(g, list) else g * cfg.ihGbar
                    if secName.startswith("dend") and hasattr(cfg, "ihGbarBasal"):
                        scaled = [v * cfg.ihGbarBasal for v in scaled] if isinstance(scaled, list) else scaled * cfg.ihGbarBasal
                    Ih["gIhbar"] = scaled

        # 8) Reduce dendritic Na on apical sections
        if hasattr(cfg, "dendNa"):
            for secName, sec in rule["secs"].items():
                if secName.startswith("apic"):
                    mechs = sec.get("mechs", {})
                    for mname in ("na12", "na12mut"):
                        if mname in mechs and "gbar" in mechs[mname]:
                            g = mechs[mname]["gbar"]
                            mechs[mname]["gbar"] = [v * cfg.dendNa for v in g] if isinstance(g, list) else g * cfg.dendNa

        # 9) Weight normalization
        netParams.addCellParamsWeightNorm(
            label,
            str(self.project_root / "conn" / "PT5B_full_weightNorm_TIM.pkl"),
            threshold=getattr(cfg, "weightNormThreshold", None),
        )

        # 10) Optional: save JSON snapshot of the rule
        if getattr(cfg, "saveCellParams", False):
            netParams.saveCellParamsRule(
                label=label,
                fileName=str(self.project_root / "cells" / "Na12HH16HH_TF.json"),
            )
        rule = netParams.cellParams["PT5B_full"]
        param_path = str(
            self.project_root / "cells" / "Neuron_HH_Adult-main" / "Neuron_Model_12HH16HH" / "params" / "na12annaTFHH2mut.txt"
        )
        try:
            apply_na_paramfile_to_rule(rule, param_path)
        except FileNotFoundError:
            pass

    # ------------------------------ ImportSpec --------------------------------

    def import_spec(self, ctx) -> ImportSpec:
        """
        Prepare the ImportSpec for PT5B_full (Tim variant):
          - Write JSON params the model reads (from CSV + cfg.variant/cfg.loadmutantParams)
          - Import from Python source: Na12HMMModel_TF.py / Na12Model_TF
          - Save/load path points to JSON so subsequent runs can load directly
        """
        self.ctx = ctx

        label = "PT5B_full"
        file_py = self.project_root / "cells" / "Neuron_HH_Adult-main" / "Na12HMMModel_TF.py"
        json_params_path = (
            self.project_root
            / "cells"
            / "Neuron_HH_Adult-main"
            / "Neuron_Model_12HH16HH"
            / "params"
            / "na12annaTFHH2mut.txt"
        )
        model_json_rule = self.project_root / "cells" / "Na12HH16HH_TF.json"

        # 1) CSV -> JSON parameter write
        variant = ctx.cfg.variant if getattr(ctx.cfg, "loadmutantParams", False) else "WT"
        csv_path = self.project_root / "cells" / "Neuron_HH_Adult-main" / "MutantParameters_updated_062725.csv"
        variants = csv_to_dict(str(csv_path))
        sorted_variant = dict(sorted(variants[variant].items()))
        print("Writing Na12 model params JSON for variant:", variant, sorted_variant)
        

        for k, v in sorted_variant.items():
            sorted_variant[k] = float(v)
            try:
                sorted_variant[k] = float(v)
            except Exception:
                # keep as string if not convertible
                pass
        # json_params_path.parent.mkdir(parents=True, exist_ok=True)
        # with open(json_params_path, "w") as f:
        #     json.dump(sorted_variant, f)

        json_params_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(json_params_path, sorted_variant)

        # 2) Build ImportSpec
        conds: Dict[str, Any] = {"cellType": "PT", "cellModel": "HH_full"}

        return ImportSpec(
            label=label,
            conds=conds,
            kind="python",
            file=file_py,
            cell_name="Na12Model_TF",
            kwargs={
                "cellInstance": False,  # importing a rule
                "somaAtOrigin": True,
            },
            # Use JSON for persistence/loading
            save_to_pkl=model_json_rule,
            load_from_pkl=model_json_rule,
            post_fn=self._post,
        )
