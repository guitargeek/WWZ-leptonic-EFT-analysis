import uproot
import uproot_methods
import numpy as np

import os


scale_factor_path = "resources/scalefactors/wvz/v1"


class EGammaScaleFactors(object):
    def __init__(self, file_name):
        f = uproot.open(os.path.join(scale_factor_path, file_name))
        h = f["EGamma_SF2D"]
        self.df_ = h.pandas()

    def __call__(self, pt, eta):

        if hasattr(pt, "awkward"):
            return pt.awkward.JaggedArray.fromcounts(pt.counts, self(pt.flatten(), eta.flatten()))

        indices = zip(eta, np.minimum(pt, 499.9))
        return self.df_.loc[indices]["count"].values


elec_reco_highpt_sf = {
    2016: EGammaScaleFactors("EGM2D_BtoH_GT20GeV_RecoSF_Legacy2016.root"),
    2017: EGammaScaleFactors("egammaEffi.txt_EGM2D_runBCDEF_passingRECO.root"),
}

elec_reco_lowpt_sf = {
    2016: EGammaScaleFactors("EGM2D_BtoH_low_RecoSF_Legacy2016.root"),
    2017: EGammaScaleFactors("egammaEffi.txt_EGM2D_runBCDEF_passingRECO_lowEt.root"),
}

elec_reco_sf = {
    2016: lambda pt, eta: (pt <= 20.0) * elec_reco_lowpt_sf[2016](pt, eta)
    + (pt > 20.0) * elec_reco_highpt_sf[2016](pt, eta),
    2017: lambda pt, eta: (pt <= 20.0) * elec_reco_lowpt_sf[2017](pt, eta)
    + (pt > 20.0) * elec_reco_highpt_sf[2017](pt, eta),
    2018: EGammaScaleFactors("egammaEffi.txt_EGM2D_updatedAll.root"),
}

elec_medium_sf = {
    2016: EGammaScaleFactors("2016LegacyReReco_ElectronMedium_Fall17V2.root"),
    2017: EGammaScaleFactors("2017_ElectronMedium.root"),
    2018: EGammaScaleFactors("2018_ElectronMedium.root"),
}

elec_veto_sf = {
    2016: EGammaScaleFactors("2016_ElectronWPVeto_Fall17V2.root"),
    2017: EGammaScaleFactors("2017_ElectronWPVeto_Fall17V2.root"),
    2018: EGammaScaleFactors("2018_ElectronWPVeto_Fall17V2.root"),
}

elec_mva_medium_sf = {
    2016: EGammaScaleFactors("2016LegacyReReco_ElectronMVA90_Fall17V2.root"),
    2017: EGammaScaleFactors("2017_ElectronMVA90.root"),
    2018: EGammaScaleFactors("2018_ElectronMVA90.root"),
}


class MuonScaleFactors(object):
    def __init__(self, file_name, hist_name, index_names=None):
        f = uproot.open(os.path.join(scale_factor_path, file_name))
        h = f[hist_name]
        self.df_ = h.pandas()

        if index_names is None:
            self.index_names_ = self.df_.index.names
        else:
            self.index_names_ = index_names

        if self.index_names_ == ["pt", "abseta"]:
            pass
        elif self.index_names_ == ["pt", "eta"]:
            pass
        elif self.index_names_ == ["eta", "pt"]:
            pass
        elif self.index_names_ == ["pt", "eta"]:
            pass
        else:
            raise ValueError(
                "Problem with " + file_name + ":" + hist_name + ". Don't know how to pass eta and pt from axis names!"
            )

    def __call__(self, pt, eta):

        if hasattr(pt, "awkward"):
            return pt.awkward.JaggedArray.fromcounts(pt.counts, self(pt.flatten(), eta.flatten()))

        pt = np.minimum(pt, 119.9)

        if self.index_names_ == ["pt", "abseta"]:
            indices = zip(pt, np.abs(eta))
        elif self.index_names_ == ["abseta", "pt"]:
            indices = zip(np.abs(eta), pt)
        elif self.index_names_ == ["eta", "pt"]:
            indices = zip(eta, pt)
        elif self.index_names_ == ["pt", "eta"]:
            indices = zip(pt, eta)

        return self.df_.loc[indices]["count"].values


muon_BCDEF_id_sf_2016_ = MuonScaleFactors(
    "EfficiencyStudies_2016_rootfiles_RunBCDEF_SF_ID.root",
    "NUM_MediumID_DEN_genTracks_eta_pt",
    index_names=["eta", "pt"],
)
muon_GH_id_sf_2016_ = MuonScaleFactors(
    "EfficiencyStudies_2016_rootfiles_RunGH_SF_ID.root", "NUM_MediumID_DEN_genTracks_eta_pt", index_names=["eta", "pt"]
)

muon_id_sf = {
    2016: lambda pt, eta: 0.55 * muon_BCDEF_id_sf_2016_(pt, eta) + 0.45 * muon_GH_id_sf_2016_(pt, eta),
    2017: MuonScaleFactors(
        "EfficiencyStudies_2017_rootfiles_RunBCDEF_SF_ID.root", "NUM_MediumID_DEN_genTracks_pt_abseta"
    ),
    2018: MuonScaleFactors(
        "EfficiencyStudies_2018_rootfiles_RunABCD_SF_ID.root", "NUM_MediumID_DEN_TrackerMuons_pt_abseta"
    ),
}

muon_BCDEF_id_lowpt_sf_2016_ = MuonScaleFactors(
    "EfficiencyStudies_2016_rootfiles_lowpt_RunBCDEF_SF_ID.root", "NUM_MediumID_DEN_genTracks_pt_abseta"
)
muon_GH_id_lowpt_sf_2016_ = MuonScaleFactors(
    "EfficiencyStudies_2016_rootfiles_lowpt_RunGH_SF_ID.root", "NUM_MediumID_DEN_genTracks_pt_abseta"
)

muon_id_lowpt_sf = {
    2016: lambda pt, eta: 0.55 * muon_BCDEF_id_lowpt_sf_2016_(pt, eta) + 0.45 * muon_GH_id_lowpt_sf_2016_(pt, eta),
    2017: MuonScaleFactors(
        "EfficiencyStudies_2017_rootfiles_lowpt_RunBCDEF_SF_ID_JPsi.root", "NUM_MediumID_DEN_genTracks_pt_abseta"
    ),
    2018: MuonScaleFactors(
        "EfficiencyStudies_2018_rootfiles_lowpt_RunABCD_SF_ID.root", "NUM_MediumID_DEN_genTracks_pt_abseta"
    ),
}

muon_reco_sf = {
    2016: lambda pt, eta: (pt <= 20.0) * muon_id_lowpt_sf[2016](pt, eta) + (pt > 20.0) * muon_id_sf[2016](pt, eta),
    2017: lambda pt, eta: (pt <= 20.0) * muon_id_lowpt_sf[2017](pt, eta) + (pt > 20.0) * muon_id_sf[2017](pt, eta),
    2018: lambda pt, eta: (pt <= 20.0) * muon_id_lowpt_sf[2018](pt, eta) + (pt > 20.0) * muon_id_sf[2018](pt, eta),
}

muon_BCDEF_tightiso_sf_2016_ = MuonScaleFactors(
    "EfficiencyStudies_2016_rootfiles_RunBCDEF_SF_ISO.root",
    "NUM_TightRelIso_DEN_MediumID_eta_pt",
    index_names=["eta", "pt"],
)
muon_GH_tightiso_sf_2016_ = MuonScaleFactors(
    "EfficiencyStudies_2016_rootfiles_RunGH_SF_ISO.root",
    "NUM_TightRelIso_DEN_MediumID_eta_pt",
    index_names=["eta", "pt"],
)

muon_tightiso_sf = {
    2016: lambda pt, eta: 0.55 * muon_BCDEF_tightiso_sf_2016_(pt, eta) + 0.45 * muon_GH_tightiso_sf_2016_(pt, eta),
    2017: MuonScaleFactors(
        "EfficiencyStudies_2017_rootfiles_RunBCDEF_SF_ISO.root", "NUM_TightRelIso_DEN_MediumID_pt_abseta"
    ),
    2018: MuonScaleFactors(
        "EfficiencyStudies_2018_rootfiles_RunABCD_SF_ISO.root", "NUM_TightRelIso_DEN_MediumID_pt_abseta"
    ),
}

muon_BCDEF_looseiso_sf_2016_ = MuonScaleFactors(
    "EfficiencyStudies_2016_rootfiles_RunBCDEF_SF_ISO.root",
    "NUM_LooseRelIso_DEN_MediumID_eta_pt",
    index_names=["eta", "pt"],
)
muon_GH_looseiso_sf_2016_ = MuonScaleFactors(
    "EfficiencyStudies_2016_rootfiles_RunGH_SF_ISO.root",
    "NUM_LooseRelIso_DEN_MediumID_eta_pt",
    index_names=["eta", "pt"],
)

muon_looseiso_sf = {
    2016: lambda pt, eta: 0.55 * muon_BCDEF_looseiso_sf_2016_(pt, eta) + 0.45 * muon_GH_looseiso_sf_2016_(pt, eta),
    2017: MuonScaleFactors(
        "EfficiencyStudies_2017_rootfiles_RunBCDEF_SF_ISO.root", "NUM_LooseRelIso_DEN_MediumID_pt_abseta"
    ),
    2018: MuonScaleFactors(
        "EfficiencyStudies_2018_rootfiles_RunABCD_SF_ISO.root", "NUM_LooseRelIso_DEN_MediumID_pt_abseta"
    ),
}
