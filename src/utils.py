from calendar import month_name

from src import constants


def get_month_season_mapping():
    # map month number to str of abbreviations of month names
    # month number refers to the last month of the season
    return {
        m: "".join(
            [
                month_name[(m - i) % 12 + 1][0]
                for i in range(constants.seas_len, 0, -1)
            ]
        )
        for m in range(1, 13)
    }
