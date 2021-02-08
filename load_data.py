import pandas as pd
from os import listdir
from os.path import isfile, join

def load_abs_surfs():
    abs_surfs = pd.read_excel('data/absSurfs.xlsx').set_index("Site")
    abs_surfs = abs_surfs.stack()
    abs_surfs.index.names = ["Site", "CLC"]
    idx = abs_surfs.index
    abs_surfs.index = abs_surfs.index.set_levels([idx.levels[0].to_series(), idx.levels[1].to_series().astype(int)])
    abs_surfs = pd.DataFrame({'surface': abs_surfs})
    return abs_surfs

def load_pesticides():
    pesticides = pd.read_excel('data/pesticides.xlsx').rename(columns={'importName': 'pesticide'})
    pesticides = pesticides.set_index("pesticide")
    pesticides["typeEN"] = pesticides["typeEN"].str.lower().str.replace("and", ",",regex=True).str.replace(" +", "",regex=True)
    pesticides["familyEN"] = pesticides["familyEN"].str.lower().str.replace(" +", ",",regex=True)
    return pesticides

def load_libelles():
    a = pd.read_excel('data/clc-nomenclature-c.xls', sheet_name="nomenclature_clc_1").rename(columns={'code_clc_1': "CLC"})
    b = pd.read_excel('data/clc-nomenclature-c.xls', sheet_name="nomenclature_clc_2").rename(columns={'code_clc_2': "CLC"})
    c = pd.read_excel('data/clc-nomenclature-c.xls', sheet_name="nomenclature_clc_3").rename(columns={'code_clc_3': "CLC"})
    libelles = pd.concat((a, b, c)).set_index("CLC")
    return libelles

def load_disthive():
    dist_beehive = pd.read_excel('data/distsOneSheet.xlsx').rename(columns={"classCLC": "CLC"}).set_index(["Site", "polyID", "CLC"])
    return dist_beehive

def load_periods(category):
    if category not in ["HM", "Pesticides"]:
        raise ValueError("cateogry can only be HM or Pesticides")
    
    all_data = []
    for year in ["2017", "2018", "2019", "2020"]:
        year_path = f"data/{year}/{category}"
        onlyfiles = [join(year_path, f) for f in listdir(year_path) if isfile(join(year_path, f)) and f.endswith(".xlsx") and not f.startswith('~')]
        for excel in onlyfiles:
            all_data.append(pd.read_excel(excel))
    return pd.concat(all_data).rename(columns={"REF....SUBSTANCE": "Site", "PERIOD": "Period"}).set_index(["Site", "Period"])

def load_heavy_metal_lmr():
    lmr = pd.read_csv("data/LMR.txt", sep="\t").drop("Unnamed: 7", axis=1).stack().droplevel(0)
    lmr.index.name = "heavymetal"
    lmr = pd.DataFrame({'LMR': lmr})
    return lmr

def get_pesticides_flags(phm_grouped, pesticide_cat, pesticides_family):
    df = phm_grouped['pesticide'].stack()
    # Pesticides
    df.index.names = ["Site", "pesticide"]
    site_per_pesticide = df.reset_index().rename(columns={0: 'level'}).set_index('pesticide').merge(pesticides[['LMR']], left_index=True, right_index=True)
    site_per_pesticide = site_per_pesticide.assign(
        above_LMR=site_per_pesticide['level'] > site_per_pesticide['LMR'],
        present=site_per_pesticide['level'] > 0
    )

    pesticide_cat_per_site = (site_per_pesticide.merge(pesticide_cat, left_index=True, right_index=True).groupby(['Site', 'pesticide_cat']).max())
    pesticide_cat_per_site = pesticide_cat_per_site.unstack()

    pesticide_fam_per_site = (site_per_pesticide.merge(pesticides_family, left_index=True, right_index=True).groupby(['Site', 'pesticide_family']).max())
    pesticide_fam_per_site = pesticide_fam_per_site.unstack()

    site_per_pesticide = site_per_pesticide.reset_index().set_index(['Site', 'pesticide'])

    pesticides_flags = pd.concat(
        [
            site_per_pesticide.unstack(),
            pesticide_cat_per_site,
            pesticide_fam_per_site,
        ],
        axis=1,
        keys=['pesticide', 'pesticide_cat','pesticide_family'],
    )
    pesticides_flags

    return pesticides_flags

def get_heavymetal_flags(phm_grouped):
    df = phm_grouped['heavymetal'].stack()
    # Pesticides
    df.index.names = ["Site", "heavymetal"]
    df = df.reset_index().rename(columns={0: 'level'}).set_index('heavymetal')
    df = df.merge(heavy_metal_lmr, left_index=True, right_index=True).reset_index().set_index(['Site', 'heavymetal'])
    df = df.assign(
        above_LMR=df['level'] > df['LMR'],
        present=df['level'] > 0
    )
    df = df.unstack()
    # Add column level
    df = pd.concat([df], axis=1, keys=['heavymetal'])
    return df

def get_phm_flags(phm_grouped, pesticide_cat, pesticides_family):
    df = pd.concat([
        get_pesticides_flags(phm_grouped, pesticide_cat, pesticides_family),
        get_heavymetal_flags(phm_grouped),
    ], axis=1)
    # Drop LMR as it's not useful
    df = df.drop(['LMR'], axis=1, level=1)
    return df

# Load all the data
heavy_metal_lmr = load_heavy_metal_lmr()
abs_surfs = load_abs_surfs() # Merged later
pesticides = load_pesticides()
libelles = load_libelles()
disthive = load_disthive() # Merged later
heavy_metal_periods = load_periods("HM")
pesticides_periods = load_periods("Pesticides")

# Start merging data together

# Average the distances to the hives
distsurf = disthive.groupby(['Site', 'CLC']).mean().merge(abs_surfs, left_index=True, right_index=True)

# Load historical data
phm = pd.concat(
    [pesticides_periods,
    heavy_metal_periods],
    axis=1,
    keys=['pesticide','heavymetal'],
)

# Taking the max of the historical measurements
phm_grouped = phm.groupby('Site').max().fillna(0)

# Categorize pesticides
pesticides_family = pd.DataFrame({'pesticide_family': pesticides[["familyEN"]].fillna("UNKNOWN").apply(lambda r: r[0].split(','), axis=1).explode()})
pesticide_cat = pd.DataFrame({'pesticide_cat': pesticides[["typeEN"]].fillna("UNKNOWN").apply(lambda r: r[0].split(','), axis=1).explode()})

# Pesticide and heavy metals flags
phm_flags = get_phm_flags(phm_grouped, pesticide_cat, pesticides_family)

# We've now got basic features and basic things to predict.
# Next steps are to engineer these features, have a look at them and select a subset from the things to predict
features = distsurf.unstack().fillna(0)

# Contains the level, above_LMR, present for the pesticides, pesticides_category, pesticides_family, heavy_metal
to_predict = phm_flags

print("Features categories: " + str(features.columns.get_level_values(0).unique()))

print("To predict categories: " + str(to_predict.columns.get_level_values(0).unique()))
print("To predict subcategories: " + str(to_predict.columns.get_level_values(1).unique()))

display(to_predict)
display(features)