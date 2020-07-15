# Helper classes


class YearDependent(object):
    def __init__(self, parameters):
        self.params = parameters


class Configuration(object):
    def __init__(self, **params):
        self.params_ = params

    def __getitem__(self, key):
        if year not in (2016, 2017, 2018):
            raise RuntimeError("year is not configured correctly!")
        if key in self.params_ and type(self.params_[key]) == YearDependent:
            return self.params_[key].params[year]
        return self.params_[key]


# Actual configuration goes here

year = None

cfg = Configuration(
    # jet id: loose for 2016, tight for 2018 and 2019
    jet_id_min_nConstituents=YearDependent({2016: 2, 2017: 2, 2018: 2}),
    jet_id_max_neHEF=YearDependent({2016: 0.99, 2017: 0.9, 2018: 0.9}),
    jet_id_max_neEmEF=YearDependent({2016: 0.99, 2017: 0.9, 2018: 0.9}),
    jet_id_min_chHEF=YearDependent({2016: 0.0, 2017: 0.0, 2018: 0.0}),
    jet_id_max_chEmEF=YearDependent({2016: 0.99, 2017: 1.0, 2018: 1.0}),
    jet_id_min_ch_nConstituents=YearDependent({2016: 0, 2017: 0, 2018: 0}),
    # for jet and bjet selection
    jet_max_eta=2.4,
    jet_min_pt=30.0,
    b_jet_min_pt=20.0,
    jet_btagDeepB_cut=YearDependent({2016: 0.2217, 2017: 0.1522, 2018: 0.1241}),
)
