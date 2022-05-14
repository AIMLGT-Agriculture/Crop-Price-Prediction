import pandas as pd
import numpy as np
import numpy as np
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import sklearn.model_selection as xval
from sklearn.datasets import fetch_openml
import forestci as fci
from sklearn import metrics
from sklearn.metrics import r2_score
import statistics
import matplotlib.pyplot as plt 


# Functions to implement soft impute
#these are taken from Someshwar's work


def getNumberofDays(year):
    p = pd.Period(f'{year}-{1}-1')
    number_of_days = p.is_leap_year+365
    #number_of_days = sum([pd.Period(f'{year}-{i}-1').daysinmonth for i in range(1,13)])
    return number_of_days


def masked_mae(X_true, X_pred, mask):
    masked_diff = X_true[mask] - X_pred[mask]
    return np.mean(np.abs(masked_diff))
def generate_random_column_samples(column):
    col_mask = np.isnan(column)
    n_missing = np.sum(col_mask)
    if n_missing == len(column):
        #logging.warn("No observed values in column")
        return np.zeros_like(column)

    mean = np.nanmean(column)
    std = np.nanstd(column)

    if np.isclose(std, 0):
        return np.array([mean] * n_missing)
    else:
        return np.random.randn(n_missing) * std + mean
    
    
import warnings
from sklearn.utils import check_array

class Solver(object):
    def __init__(
            self,
            fill_method="zero",
            min_value=None,
            max_value=None,
            normalizer=None):
        self.fill_method = fill_method
        self.min_value = min_value
        self.max_value = max_value
        self.normalizer = normalizer

    def __repr__(self):
        return str(self)

    def __str__(self):
        field_list = []
        for (k, v) in sorted(self.__dict__.items()):
            if v is None or isinstance(v, (float, int)):
                field_list.append("%s=%s" % (k, v))
            elif isinstance(v, str):
                field_list.append("%s='%s'" % (k, v))
        return "%s(%s)" % (
            self.__class__.__name__,
            ", ".join(field_list))

    def _check_input(self, X):
        if len(X.shape) != 2:
            raise ValueError("Expected 2d matrix, got %s array" % (X.shape,))

    def _check_missing_value_mask(self, missing):
        if not missing.any():
            warnings.simplefilter("always")
            warnings.warn("Input matrix is not missing any values")
        if missing.all():
            raise ValueError("Input matrix must have some non-missing values")

    def _fill_columns_with_fn(self, X, missing_mask, col_fn):
        for col_idx in range(X.shape[1]):
            missing_col = missing_mask[:, col_idx]
            n_missing = missing_col.sum()
            if n_missing == 0:
                continue
            col_data = X[:, col_idx]
            fill_values = col_fn(col_data)
            if np.all(np.isnan(fill_values)):
                fill_values = 0
            X[missing_col, col_idx] = fill_values

    def fill(
            self,
            X,
            missing_mask,
            fill_method=None,
            inplace=False):
        """
        Parameters
        ----------
        X : np.array
            Data array containing NaN entries
        missing_mask : np.array
            Boolean array indicating where NaN entries are
        fill_method : str
            "zero": fill missing entries with zeros
            "mean": fill with column means
            "median" : fill with column medians
            "min": fill with min value per column
            "random": fill with gaussian samples according to mean/std of column
        inplace : bool
            Modify matrix or fill a copy
        """
        X = check_array(X, force_all_finite=False)

        if not inplace:
            X = X.copy()

        if not fill_method:
            fill_method = self.fill_method

        if fill_method not in ("zero", "mean", "median", "min", "random"):
            raise ValueError("Invalid fill method: '%s'" % (fill_method))
        elif fill_method == "zero":
            # replace NaN's with 0
            X[missing_mask] = 0
        elif fill_method == "mean":
            self._fill_columns_with_fn(X, missing_mask, np.nanmean)
        elif fill_method == "median":
            self._fill_columns_with_fn(X, missing_mask, np.nanmedian)
        elif fill_method == "min":
            self._fill_columns_with_fn(X, missing_mask, np.nanmin)
        elif fill_method == "random":
            self._fill_columns_with_fn(
                X,
                missing_mask,
                col_fn=generate_random_column_samples)
        return X

    def prepare_input_data(self, X):
        """
        Check to make sure that the input matrix and its mask of missing
        values are valid. Returns X and missing mask.
        """
        X = check_array(X, force_all_finite=False)
        if X.dtype != "f" and X.dtype != "d":
            X = X.astype(float)

        self._check_input(X)
        missing_mask = np.isnan(X)
        self._check_missing_value_mask(missing_mask)
        return X, missing_mask

    def clip(self, X):
        """
        Clip values to fall within any global or column-wise min/max constraints
        """
        X = np.asarray(X)
        if self.min_value is not None:
            X[X < self.min_value] = self.min_value
        if self.max_value is not None:
            X[X > self.max_value] = self.max_value
        return X

    def project_result(self, X):
        """
        First undo normalization and then clip to the user-specified min/max
        range.
        """
        X = np.asarray(X)
        if self.normalizer is not None:
            X = self.normalizer.inverse_transform(X)
        return self.clip(X)

    def solve(self, X, missing_mask):
        """
        Given an initialized matrix X and a mask of where its missing values
        had been, return a completion of X.
        """
        raise ValueError("%s.solve not yet implemented!" % (
            self.__class__.__name__,))

    def fit_transform(self, X, y=None):
        """
        Fit the imputer and then transform input `X`
        Note: all imputations should have a `fit_transform` method,
        but only some (like IterativeImputer in sklearn) also support inductive
        mode using `fit` or `fit_transform` on `X_train` and then `transform`
        on new `X_test`.
        """
        X_original, missing_mask = self.prepare_input_data(X)
        observed_mask = ~missing_mask
        X = X_original.copy()
        if self.normalizer is not None:
            X = self.normalizer.fit_transform(X)
        X_filled = self.fill(X, missing_mask, inplace=True)
        if not isinstance(X_filled, np.ndarray):
            raise TypeError(
                "Expected %s.fill() to return NumPy array but got %s" % (
                    self.__class__.__name__,
                    type(X_filled)))

        X_result = self.solve(X_filled, missing_mask)
        if not isinstance(X_result, np.ndarray):
            raise TypeError(
                "Expected %s.solve() to return NumPy array but got %s" % (
                    self.__class__.__name__,
                    type(X_result)))

        X_result = self.project_result(X=X_result)
        X_result[observed_mask] = X_original[observed_mask]
        return X_result

    def fit(self, X, y=None):
        """
        Fit the imputer on input `X`.
        Note: all imputations should have a `fit_transform` method,
        but only some (like IterativeImputer in sklearn) also support inductive
        mode using `fit` or `fit_transform` on `X_train` and then `transform`
        on new `X_test`.
        """
        raise ValueError(
            "%s.fit not implemented! This imputation algorithm likely "
            "doesn't support inductive mode. Only fit_transform is "
            "supported at this time." % (
                self.__class__.__name__,))

    def transform(self, X, y=None):
        """
        Transform input `X`.
        Note: all imputations should have a `fit_transform` method,
        but only some (like IterativeImputer in sklearn) also support inductive
        mode using `fit` or `fit_transform` on `X_train` and then `transform`
        on new `X_test`.
        """
        raise ValueError(
            "%s.transform not implemented! This imputation algorithm likely "
            "doesn't support inductive mode. Only %s.fit_transform is "
            "supported at this time." % (
                self.__class__.__name__, self.__class__.__name__))
        
        
from sklearn.utils.extmath import randomized_svd
from sklearn.utils import check_array

F32PREC = np.finfo(np.float32).eps


class SoftImpute(Solver):
    """
    Implementation of the SoftImpute algorithm from:
    "Spectral Regularization Algorithms for Learning Large Incomplete Matrices"
    by Mazumder, Hastie, and Tibshirani.
    """
    def __init__(
            self,
            shrinkage_value=None,
            convergence_threshold=0.001,
            max_iters=100,
            max_rank=None,
            n_power_iterations=1,
            init_fill_method="zero",
            min_value=None,
            max_value=None,
            normalizer=None,
            verbose=True):
        """
        Parameters
        ----------
        shrinkage_value : float
            Value by which we shrink singular values on each iteration. If
            omitted then the default value will be the maximum singular
            value of the initialized matrix (zeros for missing values) divided
            by 50.
        convergence_threshold : float
            Minimum ration difference between iterations (as a fraction of
            the Frobenius norm of the current solution) before stopping.
        max_iters : int
            Maximum number of SVD iterations
        max_rank : int, optional
            Perform a truncated SVD on each iteration with this value as its
            rank.
        n_power_iterations : int
            Number of power iterations to perform with randomized SVD
        init_fill_method : str
            How to initialize missing values of data matrix, default is
            to fill them with zeros.
        min_value : float
            Smallest allowable value in the solution
        max_value : float
            Largest allowable value in the solution
        normalizer : object
            Any object (such as BiScaler) with fit() and transform() methods
        verbose : bool
            Print debugging info
        """
        Solver.__init__(
            self,
            fill_method=init_fill_method,
            min_value=min_value,
            max_value=max_value,
            normalizer=normalizer)
        self.shrinkage_value = shrinkage_value
        self.convergence_threshold = convergence_threshold
        self.max_iters = max_iters
        self.max_rank = max_rank
        self.n_power_iterations = n_power_iterations
        self.verbose = verbose

    def _converged(self, X_old, X_new, missing_mask):
        # check for convergence
        old_missing_values = X_old[missing_mask]
        new_missing_values = X_new[missing_mask]
        difference = old_missing_values - new_missing_values
        ssd = np.sum(difference ** 2)
        old_norm = np.sqrt((old_missing_values ** 2).sum())
        # edge cases
        if old_norm == 0 or (old_norm < F32PREC and np.sqrt(ssd) > F32PREC):
            return False
        else:
            return (np.sqrt(ssd) / old_norm) < self.convergence_threshold

    def _svd_step(self, X, shrinkage_value, max_rank=None):
        """
        Returns reconstructed X from low-rank thresholded SVD and
        the rank achieved.
        """
        if max_rank:
            # if we have a max rank then perform the faster randomized SVD
            (U, s, V) = randomized_svd(
                X,
                max_rank,
                n_iter=self.n_power_iterations)
        else:
            # perform a full rank SVD using ARPACK
            (U, s, V) = np.linalg.svd(
                X,
                full_matrices=False,
                compute_uv=True)
        s_thresh = np.maximum(s - shrinkage_value, 0)
        rank = (s_thresh > 0).sum()
        s_thresh = s_thresh[:rank]
        U_thresh = U[:, :rank]
        V_thresh = V[:rank, :]
        S_thresh = np.diag(s_thresh)
        X_reconstruction = np.dot(U_thresh, np.dot(S_thresh, V_thresh))
        return X_reconstruction, rank

    def _max_singular_value(self, X_filled):
        # quick decomposition of X_filled into rank-1 SVD
        _, s, _ = randomized_svd(
            X_filled,
            1,
            n_iter=5)
        return s[0]

    def solve(self, X, missing_mask):
        X = check_array(X, force_all_finite=False)

        X_init = X.copy()

        X_filled = X
        observed_mask = ~missing_mask
        max_singular_value = self._max_singular_value(X_filled)
        if self.verbose:
            print("[SoftImpute] Max Singular Value of X_init = %f" % (
                max_singular_value))

        if self.shrinkage_value:
            shrinkage_value = self.shrinkage_value
        else:
            # totally hackish heuristic: keep only components
            # with at least 1/50th the max singular value
            shrinkage_value = max_singular_value / 50.0

        for i in range(self.max_iters):
            X_reconstruction, rank = self._svd_step(
                X_filled,
                shrinkage_value,
                max_rank=self.max_rank)
            X_reconstruction = self.clip(X_reconstruction)

            # print error on observed data
            if self.verbose:
                mae = masked_mae(
                    X_true=X_init,
                    X_pred=X_reconstruction,
                    mask=observed_mask)
                print(
                    "[SoftImpute] Iter %d: observed MAE=%0.6f rank=%d" % (
                        i + 1,
                        mae,
                        rank))

            converged = self._converged(
                X_old=X_filled,
                X_new=X_reconstruction,
                missing_mask=missing_mask)
            X_filled[missing_mask] = X_reconstruction[missing_mask]
            if converged:
                break
        if self.verbose:
            print("[SoftImpute] Stopped after iteration %d for lambda=%f" % (
                i + 1,
                shrinkage_value))

        return X_filled
    
    
# Reading arrival and price data respectively
    
    
mod_df = pd.read_csv('Tomato/Tomato/BAK/Tomato_UP_Arrival_Raw.csv')
price_df = pd.read_csv('Tomato/Tomato/BAK/Tomato_UP_Price_Raw.csv')

# dates for price data are given in formar 'dd mm yyyy'. So we are converting it to 'dd-mm-yyyy' such that it will 
# similar to the dates in arrival data set
m = price_df['Price Date']
for  i in range(len(m)):
  date = m[i][:2]+'-'+m[i][3:6]+'-'+m[i][7:]
  m[i] = date
    
price_df['Date'] = m


#Dropping columns that are not required
price_df= price_df.drop(['State','State Code','District Code','Market Code','Commodity','Commodity Code',' Variety','Grade',
                        'Min Price','Max Price','Price Date'], axis=1)



mod_df= mod_df.drop(['State','State Code','Commodity','Commodity Code'], axis=1)

# Renaming columns Volume date as date in arrival data
mod_df.rename(columns = {'Volume Date':'Date'}, inplace = True)


#Merging arrival and Price data on date,district and Market
df_new = pd.merge(mod_df,price_df,on=['Date','District','Market'])
mod_df = df_new


# mandi_list = mod_df['Market'].unique()

n = mod_df['Date'].unique()


#Sorting data by market and dates

mod_df = mod_df.sort_values(by = ['Market','Date'])
mod_df = mod_df.reset_index(drop=True)


# lists of unique districts and Markets
districts = mod_df['District'].unique()
markets = mod_df['Market'].unique()



df=mod_df


# Adding new columns as month namem, month, day, year from column date

month_array = []
day_array=[]
year_array=[]
for i in range(df.shape[0]):
    month_array.append(df["Date"][i][3:6])
    day_array.append(int(df['Date'][i][0:2]))
    year_array.append(int(df['Date'][i][7:]))
    
df['Month_name']= month_array
df['Day'] = day_array
df['Year']= year_array

month_names = np.array(df['Month_name'])
month_names = list(month_names) 
months = month_names.copy()

for i in range(len(months)):
  if month_names[i] == "Jan":
    months[i] = 1
  elif month_names[i] == "Feb":
    months[i] = 2
  elif month_names[i] == "Mar":
    months[i] = 3
  elif month_names[i] == "Apr":
    months[i] = 4
  elif month_names[i] == "May":
    months[i] = 5
  elif month_names[i] == "Jun":
    months[i] = 6
  elif month_names[i] == "Jul":
    months[i] = 7
  elif month_names[i] == "Aug":
    months[i] = 8
  elif month_names[i] == "Sep":
    months[i] = 9
  elif month_names[i] == "Oct":
    months[i] = 10
  elif month_names[i] == "Nov":
    months[i] = 11
  elif month_names[i] == "Dec":
    months[i] = 12

df['Month']=months


#Dropping data from 2008

mod_df = df.loc[df['Year'] != 2008]
mod_df = mod_df.reset_index(drop=True)


df = pd.read_csv('df_with_last_year_price_2009_2018.csv')
df = df.drop(['Unnamed: 0'],axis=1)
df = df.loc[df['Year'] != 2008]

dates = df['Date'].unique()

mod_df =mod_df.drop(['Month_name', 'Day', 'Year', 'Month'],axis=1)

mod_df = mod_df.sort_values(by = ['Market','Date'])
mod_df = mod_df.reset_index(drop=True)


import math

new_df_list=[]    # this list will contain the NAN input for the dates for which data is not available
count=0
for market in markets:
    
    cur_df = mod_df.loc[mod_df['Market'] == market]
#     print(cur_df.head())
    cur_dates = np.array(cur_df['Date'])
    district = cur_df['District'].unique()[0]
#     print(district)
#     print(cur_dates)
    for date in dates:
        if date not in cur_dates:
            new_df_list.append( [district, market,math.nan,date
                                     ,math.nan])
        else:
            count +=1
            

#Converting it to dataframe
neww_df = pd.DataFrame(new_df_list, columns=['District', 'Market', 'Arrivals', 'Date', 'Modal_Price'])


# Dropping duplicates if any
neww_df = neww_df.drop_duplicates(
  subset = ['Market', 'Date'],
  keep = 'first').reset_index(drop = True)


#Appending actual data set with the dataset having NAN for missing dates
mod_df_ = mod_df.append(neww_df)


mod_df_ = mod_df_.drop_duplicates(
  subset = ['Market', 'Date'],
  keep = 'first').reset_index(drop = True)



# making arrays for price and arrival data
arr1 = mod_df_['Modal Price'].to_numpy()
arr2 = mod_df_['Arrivals'].to_numpy()



total_days = 3652 # from 01-01-2009 to 31-12-2018
arr1 = arr1.reshape(len(mandi_list),total_days)
arr2 = arr2.reshape(len(mandi_list),total_days)



X_price_incomplete = arr1
X_arr_incomplete = arr2


year_window = 1
start_year =2009
total_years = 10 #2009-2018 



day_window_list = []
i=0
while(i<total_years):
    base_year  = start_year+i
    present_day_window = 0
    for j in range(year_window):
        if(i+j<total_years):
            year  = base_year+j
            present_day_window += getNumberofDays(year)
        else:
            break
    day_window_list.append(present_day_window)
    i += year_window

    
day_window_slice_list = []
sum_days=0
for number_days in day_window_list:
    sum_days += number_days
    day_window_slice_list.append(sum_days)
day_window_slice_list.pop(-1)


X_price_incomplete_list = np.split(X_price_incomplete,day_window_slice_list,axis=1)


price_min_value_list=[]
price_max_value_list=[]
for X_price in  X_price_incomplete_list:
    price_df = pd.DataFrame(data = X_price.reshape(-1,1))
    min_value = price_df.describe(percentiles=[0.03,0.97]).iloc[4][0]
    max_value = price_df.describe(percentiles=[0.03,0.97]).iloc[6][0]
    price_min_value_list.append(min_value)
    price_max_value_list.append(max_value)
    
    
X_price_filled_list = []
for i in range(len(X_price_incomplete_list)):
    X_price_filled = SoftImpute(convergence_threshold=0.001,init_fill_method="min",min_value=price_min_value_list[i],max_value=price_max_value_list[i]).fit_transform(X_price_incomplete_list[i])
    X_price_filled_list.append(X_price_filled)
    
X_price_filled_list = []
for i in range(len(X_price_incomplete_list)):
    X_price_filled = SoftImpute(convergence_threshold=0.001,init_fill_method="min",min_value=price_min_value_list[i],max_value=price_max_value_list[i]).fit_transform(X_price_incomplete_list[i])
    X_price_filled_list.append(X_price_filled)
    
    
X_price_filled = np.hstack(X_price_filled_list)


X_arr_incomplete_list = np.split(X_arr_incomplete,day_window_slice_list,axis=1)
arr_min_value_list=[]
arr_max_value_list=[]
for X_arr in  X_arr_incomplete_list:
    arr_df = pd.DataFrame(data = X_arr.reshape(-1,1))
    min_value = arr_df.describe(percentiles=[0.03,0.97]).iloc[4][0]
    max_value = arr_df.describe(percentiles=[0.03,0.97]).iloc[6][0]
    arr_min_value_list.append(min_value)
    arr_max_value_list.append(max_value)
    
    
X_arr_filled_list = []
for i in range(len(X_arr_incomplete_list)):
    X_arr_filled = SoftImpute(convergence_threshold=0.001,init_fill_method="min",min_value=arr_min_value_list[i],max_value=arr_max_value_list[i]).fit_transform(X_arr_incomplete_list[i])
    X_arr_filled_list.append(X_arr_filled)
    
X_arr_filled = np.hstack(X_arr_filled_list)


X_price_filled = X_price_filled.reshape(1,X_price_filled.size)[0]
X_arr_filled = X_arr_filled.reshape(1,X_arr_filled.size)[0]


mod_df_['Imp_Price'] = X_price_filled
mod_df_['Imp_Arrival'] = X_arr_filled
ArrivalNanFlag = np.isnan(mod_df_['Arrivals'])
PriceNanFlag = np.isnan(mod_df_['Modal_Price'])
mod_df_['PriceNanFlag'] = PriceNanFlag
mod_df_['ArrivalNanFlag'] = ArrivalNanFlag
mod_df_.info()


mod_df_ =mod_df_.drop(['Modal Price', 'Modal_Price','Arrivals'],axis=1)

mod_df_.to_csv('Tomato/Tomato/sample_post_impute.csv')
