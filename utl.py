import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns

from sklearn.impute import KNNImputer

def show_missing(df):
    """Return a Pandas dataframe describing the contents of a source dataframe including missing values."""
    
    variables = []
    dtypes = []
    count = []
    unique = []
    missing = []
    pc_missing = []
    
    for item in df.columns:
        variables.append(item)
        dtypes.append(df[item].dtype)
        count.append(len(df[item]))
        unique.append(len(df[item].unique()))
        missing.append(df[item].isna().sum())
        pc_missing.append(round((df[item].isna().sum() / len(df[item])) * 100, 2))

    output = pd.DataFrame({
        'variable': variables, 
        'dtype': dtypes,
        'count': count,
        'unique': unique,
        'missing': missing, 
        'pc_missing': pc_missing
    })    
        
    return output





def compare_imputation_methods(df, variables, m):
    """
    This function plots a pair of graphs comparing the original distrubution of the data and the distribution after apply missing values imputation with 3 methods: mean, mode, and KNN imputer
    """

    alternative_mean = df.copy(deep=True)
    alternative_mode = df.copy(deep=True)
    alternative_knn = df.copy(deep=True)

    imputer = KNNImputer(n_neighbors=10)

    if m == 'mean':
        for v in variables:
            alternative_mean[v] =alternative_mean[v].fillna(alternative_mean[v].mean())

            fig_mean, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
            sns.histplot(alternative_mean[v], ax = axes[0], kde=True, color='#be29ec')
            axes[0].set_title(v + ' Distribution After Mean Imputer')
            sns.histplot(df[v], ax = axes[1], kde = True, color='#be29ec')
            axes[1].set_title('Original ' + v + ' Distribution')
            plt.suptitle(v + ' Distribution Comparation with Mean Imputer', fontsize=16)
            plt.tight_layout()
    elif m == 'mode':
        for v in variables:
            alternative_mode[v] =alternative_mean[v].fillna(alternative_mean[v].mode()[0])

            fig_mode, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
            sns.histplot(alternative_mode[v], ax = axes[0], kde=True, color='#be29ec')
            axes[0].set_title(v + ' Distribution After Mode Imputer')
            sns.histplot(df[v], ax = axes[1], kde = True, color='#be29ec')
            axes[1].set_title('Original ' + v + ' Distribution')
            plt.suptitle(v + ' Distribution Comparation with Mode Imputer', fontsize=16)
            plt.tight_layout()
    elif m == 'knn':
        alternative_knn[variables] = imputer.fit_transform(alternative_knn[variables])
        for v in variables:

            fig_knn, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
            sns.histplot(alternative_knn[v], ax = axes[0], kde=True, color='#be29ec')
            axes[0].set_title(v + ' Distribution After KNN Imputer')
            sns.histplot(df[v], ax = axes[1], kde = True, color='#be29ec')
            axes[1].set_title('Original ' + v + ' Distribution')
            plt.suptitle(v + ' Distribution Comparation with KNN Imputer, n_neighbors=10', fontsize=16)
            plt.tight_layout()


# def hist_with_sd(Df: pd.DataFrame, v:str):

#     mean_v = Df[v].mean()
#     std_v = Df[v].std()
    
#     sns.histplot(Df[v], kde=True, bins=30, color='#d896ff')
#     plt.axvline(mean_v, color='#262626', linestyle='dashed', linewidth=2, label='Mean')
#     plt.axvline(mean_v + 2 * std_v, color='#944dd3', linestyle='dashed', linewidth=2)
#     plt.axvline(mean_v + std_v, color='#4f81bd', linestyle='dashed', linewidth=2)
#     plt.axvline(mean_v - std_v, color='#4f81bd', linestyle='dashed', linewidth=2)
#     plt.axvline(mean_v - 2 * std_v, color='#944dd3', linestyle='dashed', linewidth=2)
#     plt.title('Histogram of {} with SD'.format(v))
#     plt.legend()
#     plt.show()


def hist_with_sd(Df: pd.DataFrame, v:str, f:str ):

    mean_v = Df[v].mean()
    std_v = Df[v].std()
    mean_f = Df[f].mean()
    std_f = Df[f].std()

    fig_mean, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
    
    sns.histplot(Df[f], kde=True, bins=30, color='#be29ec', ax=axes[1])
    axes[1].axvline(mean_f, color='#262626', linestyle='dashed', linewidth=2, label='Mean')
    axes[1].axvline(mean_f + 2 * std_f, color='#944dd3', linestyle='dashed', linewidth=2)
    axes[1].axvline(mean_f + std_f, color='#4f81bd', linestyle='dashed', linewidth=2)
    axes[1].axvline(mean_f - std_f, color='#4f81bd', linestyle='dashed', linewidth=2)
    axes[1].axvline(mean_f - 2 * std_f, color='#944dd3', linestyle='dashed', linewidth=2)
    axes[1].set_title('Histogram of {} with SD'.format(f))
    
    sns.histplot(Df[v], ax = axes[0], kde = True, color='#be29ec')
    axes[0].axvline(mean_v, color='#262626', linestyle='dashed', linewidth=2, label='Mean')
    axes[0].axvline(mean_v + 2 * std_v, color='#944dd3', linestyle='dashed', linewidth=2)
    axes[0].axvline(mean_v + std_v, color='#4f81bd', linestyle='dashed', linewidth=2)
    axes[0].axvline(mean_v - std_v, color='#4f81bd', linestyle='dashed', linewidth=2)
    axes[0].axvline(mean_v - 2 * std_v, color='#944dd3', linestyle='dashed', linewidth=2)
    axes[0].set_title('Histogram of {} with SD'.format(v))

    plt.suptitle('Distributions Comparation Of {} and {}'.format(v,f), fontsize=16)
    plt.tight_layout()
    
