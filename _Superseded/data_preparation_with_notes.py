import polars as pl
from typing import Iterable, Self
import pandas as pd
from sklearn import model_selection
import re
import polars.selectors as cs
import xgboost as xgb


model_feats = [
    # id first and outcome
    'match_id', 'result',
    # fixed features about the match
    'year','tourney_level','surface','draw_size','best_of','round',
    'seed','entry','rank_points', 'rank_bin','opp_rank_points','opp_rank_bin','age',
    # numeric features that were simulated in previous project
    'first_in_per','first_won_per','second_won_per',
    'rally_svptw_per','ace_per','df_per',
    'first_rpw_per','second_rpw_per','rally_rpw_per',
    'bp_face_freq','ace_per_against','bp_create_freq'    
]


class TennisMatchDataset:
    """
    A pipeline to process and prepare ATP tennis match data for modeling.
    Loads raw match data from Jeff Sackmann's GitHub repository and applies transformations
    to extract player-level features and match outcomes suitable for post match win probability model.
    """

    def __init__(self,years:Iterable[int]):
        """
        Initialize the data processor with the desired range of years.

        Args:
            years (Iterable[int]): List or range of years to load match data for.
        """
        self.years = years
        self.original_df = self._load_matches()
        self.player_matches_raw = None
        self.player_matches_limited = None
        self.player_matches_wfeat = None
        self.player_matches_complete = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.train_xgbmatrix = None
        self.test_xgbmatrix = None

    def _load_matches(self)->pl.DataFrame:
        """
        Lazily load tennis match data from Jeff Sackmann's GitHub.

        Returns:
            pl.DataFrame: A polars with data pulled from GitHub.
        """
        print("Loading data...")
        match_datasets = [
            pl.scan_csv(
                f"https://raw.githubusercontent.com"+\
                f"/JeffSackmann/tennis_atp/master/atp_matches_{i}.csv",
                infer_schema_length=int(1e9)
            ) for i in self.years
        ]
        matches = pl.concat(match_datasets).collect()
        return matches
    
    def get_player_matches(self)->Self:
        """
        Reshape raw tennis data longer so that each row represents a player-match and assign to attribute.

        Returns:
            pSelf: The updated processor object.
        """
        print("Transforming to player matches...")
        matches_pl = (
            self.original_df
            .rename(
                mapping = lambda x: x if x == 'draw_size' else x.replace('1st','first')
                                    .replace('2nd','second').replace('l_','loser_').replace('w_','winner_')
            )
            .with_columns(
                match_id = pl.col('tourney_date').cast(pl.String) + '-' + 
                        pl.col('winner_id').cast(pl.String) + '-' + 
                        pl.col('loser_id').cast(pl.String),
                num_tiebreaks = pl.col('score').str.extract_all(r'\(').list.len().cast(pl.Int64),
                winner_tiebreaks_won = pl.col('score').str.extract_all(r'7-6\(').list.len(),
                loser_tiebreaks_won = pl.col('score').str.extract_all(r'6-7\(').list.len(),
                year = pl.col('tourney_date').cast(pl.String).str.slice(0,length=4).cast(pl.Int64)
            )
        )

        self.player_matches_raw = (
            matches_pl
            .unpivot(
                index = cs.exclude("^.*(winner|loser).*$")
            )
            .with_columns(
                result = pl.col('variable').str.extract('^(winner|loser)'),
                variable = pl.col('variable').str.replace('^(winner_|loser_)','')
            )
            .pivot(
                on='variable',
                values='value'
            )
        )

        return self

    def limit_player_matches(self)->Self:
        """
        Filter out edge cases such as retirements, walkovers, carpet surface, and missing values.

        Returns:
            Self: The updated processor object.
        """
        print("Limiting player matches...")
        self.player_matches_limited = (
            self.player_matches_raw
            .filter(
                ~pl.col('score').str.contains('RET|W/O|Walkover'),
                pl.col('surface')!= 'Carpet',
                pl.col(['age','rank','surface']).is_not_null(),
                pl.col('hand').is_in(['L','R']),
                ~pl.col('round').is_in(['ER','RR']) # by default, will exclude null values unless specified in nulls_equal argument
            )
        )

        return self
    
    def add_features(self)->Self:
        """
        Compute new features for both the player and their opponent for use in modeling.

        Returns:
            Self: The updated processor object.
        """
        print("Adding basic features...")
        intermediate = (
            self.player_matches_limited
            .rename(# rename columns to snake_case
                mapping=lambda x: re.sub(r'(\w)([A-Z])',r'\g<1>_\g<2>',x).lower()
            )
            .with_columns(
                rank = pl.col('rank').cast(pl.Int64),
                rally_svpt = pl.col('svpt').cast(pl.Float64) - pl.col('ace').cast(pl.Float64) - pl.col('df').cast(pl.Float64),
                seed = pl.col('seed').fill_null('Unseeded'),
                entry = pl.col('entry').fill_null('NA')
            )
            .with_columns(
                rank_bin = pl.when(pl.col('rank') <= 10)
                .then(pl.lit('Top 10'))
                .when(pl.col('rank') <= 25)
                .then(pl.lit('Top 25'))
                .when(pl.col('rank') <= 50)
                .then(pl.lit('Top 50'))
                .when(pl.col('rank') <= 100)
                .then(pl.lit('Top 100'))
                .otherwise(pl.lit('Outside Top 100')),
                tourney_level = pl.when(pl.col('tourney_level')=='M')
                .then(pl.lit('Masters'))
                .when(pl.col('tourney_level')=='A')
                .then(pl.lit('ATP 250/500'))
                .when(pl.col('tourney_level')=='G')
                .then(pl.lit('Grand Slam'))
                .when(pl.col('tourney_level')=='F')
                .then(pl.lit('Year End Final')),
                first_in_per = pl.col('first_in').cast(pl.Float64) / pl.col('svpt').cast(pl.Float64),
                first_won_per = pl.col('first_won').cast(pl.Float64) / pl.col('first_in').cast(pl.Float64),
                second_won_per = pl.col('second_won').cast(pl.Float64) / (pl.col('svpt').cast(pl.Float64) - pl.col('first_in').cast(pl.Float64)),
                rally_svptw = pl.col('first_won').cast(pl.Float64) + pl.col('second_won').cast(pl.Float64) - pl.col('ace').cast(pl.Float64),
                ace_per = pl.col('ace').cast(pl.Float64) / pl.col('svpt').cast(pl.Float64),
                df_per = pl.col('df').cast(pl.Float64) / pl.col('svpt').cast(pl.Float64),
                bp_face_freq = pl.col('bp_faced').cast(pl.Float64) / pl.col('svpt').cast(pl.Float64)
            )
            .with_columns(
                rally_svptw_per = pl.col('rally_svptw') / pl.col('rally_svpt')
            )
        )
        print("Adding opponent features...")
        opponent_stats = (
            intermediate
            .select(
                ['match_id','result','first_won_per','second_won_per',
                'rally_svptw_per','ace_per','bp_faced','svpt',
                'rank_points','rank_bin']
            )
            .with_columns(
                bp_create_freq = pl.col('bp_faced') / pl.col('svpt'),
            )
            .drop(
                ['bp_faced','svpt']
            )
            .rename(
                {
                    'first_won_per':'first_rpw_per',
                    'second_won_per':'second_rpw_per',
                    'rally_svptw_per':'rally_rpw_per',
                    'ace_per':'ace_per_against',
                    'rank_points':'opp_rank_points',
                    'rank_bin':'opp_rank_bin'
                }
            )
            .with_columns(# need to flip the result to match the opponent
                result = pl.when(pl.col('result') == 'winner').
                then(pl.lit('loser')).
                otherwise(pl.lit('winner'))
            )
        ) 

        self.player_matches_wfeat = (
            intermediate
            .join(# some matches are missing opponent stats bc the opponent might be missing rank for example
                opponent_stats,
                on = ['match_id','result'],
                how = 'inner'
            )
            # .select(model_feats)
            .drop_nulls(
                model_feats
            )
            .drop_nans(# only numeric columns whose name is a model_feat
                cs.numeric() & cs.by_name(*model_feats)
            )
        )

        return self
    
    def prepare_for_model(self)->Self:
        """
        Select model features and create dummy variables.

        Returns:
            Self: The updated processor object.
        """
        self.player_matches_complete = (
            self.player_matches_wfeat
            .select(model_feats)
            .to_dummies(
                ['surface','tourney_level','draw_size','best_of',
                'round','seed','entry','rank_bin','opp_rank_bin']
            )
            .with_columns(
                (cs.ends_with('rank_points') | cs.by_name('age')).as_expr().cast(pl.Float64),
                result = (pl.col('result')=='winner').cast(pl.Int64)
            )
        )

        return self

    def split_data(self)->Self:
        """
        Split the data into a train and test set with random seed set for reproducibility.

        Returns:
            Self: The updated processor object.
        """
        print('Splitting data into train and test data')
        self.X_train, self.X_test,self.y_train,self.y_test = model_selection.train_test_split(
            self.player_matches_complete.drop('result').to_pandas(),
            self.player_matches_complete.select('result').to_pandas(),
            test_size = 0.2,
            random_state=33
        )
        return self

    def convert_to_xgb(self)->Self:
        """
        Convert the split data into xgb.DMatrix type to be used with native XGBoost API.

        Returns:
            Self: The updated processor object
        """
        print("Converting to xgb.DMatrix")
        self.train_xgbmatrix = xgb.DMatrix(
            data = self.X_train.drop('match_id',axis=1),
            label = self.y_train
        )
        self.test_xgbmatrix = xgb.DMatrix(
            data = self.X_test.drop('match_id',axis=1),
            label = self.y_test
        )

        return self


    def get_pm_data(self)->pl.DataFrame:
        """
        Retrieve player match dataset before limiting to modeling columns and dummies.

        Returns:
            pl.DataFrame: Player match dataset.
        """
        return self.player_matches_wfeat

    def get_model_data(self)->pl.DataFrame:
        """
        Retrieve the final processed dataset suitable for model training.

        Returns:
            pl.DataFrame: The model-ready feature matrix.
        """
        return self.player_matches_complete
    
    def _repr_html_(self) -> str:
        """
        HTML representation for Jupyter notebooks.

        Returns:
            str: HTML table of the final dataframe or placeholder.
        """
        return self.player_matches_complete._repr_html_()

    
    def process(self)->None:
        """
        Run all processing steps for data to prepare for model training.
        """
        print('Processing data')
        (
            self
            .get_player_matches()
            .limit_player_matches()
            .add_features()
            .prepare_for_model()
            .split_data()
            .convert_to_xgb()
        )

test_tennis_data = TennisMatchDataset(range(2003,2024))
test_tennis_data.process()


