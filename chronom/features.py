import pandas as pd
import numpy as np

def chron_features(chronom_train, chronom_test, plavki_train, plavki_test):
        for i in ["train", "test"]:
                chronom = eval(f"chronom_{i}")
                plavki = eval(f"plavki_{i}")

                chronom = (
                        pd.read_csv("data/chronom_train.csv")
                        .drop(columns="Unnamed: 0")
                        .assign(VR_KON=pd.to_datetime(chronom.VR_KON),
                                VR_NACH=pd.to_datetime(chronom.VR_NACH))

                        .assign(VR_KON=lambda x: x.VR_KON.apply(lambda x: x.replace(year=2021) if x.year == 2011 else x),
                                VR_NACH=lambda x: x.VR_NACH.apply(lambda x: x.replace(year=2021) if x.year == 2011 else x))

                        .assign(duration=lambda x: (x.VR_KON - x.VR_NACH).dt.total_seconds(),
                                O2=lambda x: x.O2.fillna(0))
                )
                plavki = (
                        pd.read_csv("data/plavki_train.csv")
                .assign(plavka_VR_NACH=pd.to_datetime(plavki.plavka_VR_NACH),
                        plavka_VR_KON=pd.to_datetime(plavki.plavka_VR_KON))
                [["NPLV", "plavka_VR_NACH", "plavka_VR_KON"]]
                )
                df = (
                        chronom.merge(plavki, on="NPLV", how="inner")
                        .sort_values(["plavka_VR_NACH", "VR_NACH", "O2"], ascending=[True, True, False])
                        .drop_duplicates(subset=['NPLV', 'TYPE_OPER', 'NOP', 'VR_NACH', 'VR_KON'], keep="first")

                        .assign(sec_from_start=lambda x: (x.VR_NACH - x.plavka_VR_NACH).dt.total_seconds(),
                                sec_till_end=lambda x: (x.plavka_VR_KON - x.VR_KON).dt.total_seconds())

                        .query("sec_till_end > 0")

                        .assign(prep=lambda x: np.where(x.sec_from_start < 0, 1, 0))
                        .groupby(["NPLV", "TYPE_OPER", "prep"], as_index=False, sort=False)
                        
                        .agg(O2=("O2", "sum"), 
                                        total_duration=("duration", "sum"),
                                        min_duration=("duration", "min"),
                                        max_duration=("duration", "max"),
                                        total_operations=("NOP", "count"))
                        .assign(TYPE_OPER=lambda x: x.TYPE_OPER + "_" + x.prep.astype("str"))
                        .drop(columns="prep")
                        .pivot(index="NPLV", 
                                columns="TYPE_OPER",
                                values=["O2", "total_duration", "min_duration", "max_duration", "total_operations"])
                                .fillna(0)          
                )
                df.columns = df.columns.map('_'.join).str.strip('_')
                if i=="train":
                        train = df
                else:
                        test = df
        return train, test, list(train.columns), []