"""
This is a template algorithm on Quantopian for you to adapt and fill in.
"""
import quantopian.algorithm as algo
from quantopian.pipeline import Pipeline
from quantopian.pipeline.data.builtin import USEquityPricing
from quantopian.pipeline import CustomFactor
import numpy as np
from quantopian.pipeline.data import Fundamentals
from quantopian.pipeline.factors import SimpleMovingAverage
from scipy.stats.mstats import zscore

# Import built-in trading universe
from quantopian.pipeline.filters import QTradableStocksUS
from quantopian.pipeline.filters import Q500US


class Previous(CustomFactor):
    # Returns value of input x trading days ago where x is the window_length
    # Both the inputs and window_length must be specified as there are no defaults

    def compute(self, today, assets, out, inputs):
        out[:] = inputs[0]


class Fscore(CustomFactor):
    # Returns value of input x trading days ago where x is the window_length
    # Both the inputs and window_length must be specified as there are no defaults

    window_length = 1

    def compute(self, today, assets, out,
                net_income,
                roa,
                op_cash_flow,
                difference_op_cash_flow_and_net_income,
                change_long_term_debt,
                change_current_ratio,
                change_shares_issued,
                change_gross_margin,
                change_asset_turnover):
        out[:] = (net_income > 0).astype(int) + (roa > 0).astype(int) + (op_cash_flow > 0).astype(int) + (
        difference_op_cash_flow_and_net_income > 0).astype(int) + (change_long_term_debt < 0).astype(int) + (
                 change_current_ratio > 0).astype(int) + (change_shares_issued <= 0).astype(int) + (
                 change_gross_margin > 0).astype(int) + (change_asset_turnover > 0).astype(int)


def initialize(context):
    """
    Called once at the start of the algorithm.
    """
    # Rebalance on the first trading day of each week at 11AM.
    algo.schedule_function(rebalance,
                           date_rules.month_start(days_offset=0),
                           time_rules.market_open(hours=1, minutes=30))

    # # Record tracking variables at the end of each day.
    # algo.schedule_function(
    #     record_vars,
    #     algo.date_rules.every_day(),
    #     algo.time_rules.market_close(),
    # )

    # Create our dynamic stock selector.

    context.stocks_by_sector = {
        101.0: set(),
        102.0: set(),
        103.0: set(),
        205.0: set(),
        206.0: set(),
        207.0: set(),
        308.0: set(),
        309.0: set(),
        310.0: set(),
        311.0: set()
    }

    algo.attach_pipeline(make_pipeline(), 'value_pipeline')


def make_pipeline():
    # Create a reference to our trading universe
    base_universe = Q500US()

    # Get net income
    net_income = Fundamentals.net_income_cash_flow_statement.latest

    # Get operating cash flow
    op_cash_flow = Fundamentals.cash_flow_from_continuing_operating_activities.latest

    # Get return on assets
    roa = Fundamentals.roa.latest

    # Get long term debt
    long_term_debt = Fundamentals.long_term_debt.latest
    long_term_debt_1yr_ago = Previous(inputs=[Fundamentals.long_term_debt], window_length=252)

    # Get change in current ratio
    current_ratio = Fundamentals.current_ratio.latest
    current_ratio_1yr_ago = Previous(inputs=[Fundamentals.current_ratio], window_length=252)

    # Get shares issued
    share_issued = Fundamentals.share_issued.latest
    share_issued_1yr_ago = Previous(inputs=[Fundamentals.share_issued], window_length=252)

    # Get gross margin
    gross_margin = Fundamentals.gross_margin.latest
    gross_margin_1yr_ago = Previous(inputs=[Fundamentals.gross_margin], window_length=252)

    # Get asset turnover ratio
    assets_turnover = Fundamentals.assets_turnover.latest
    assets_turnover_1yr_ago = Previous(inputs=[Fundamentals.assets_turnover], window_length=252)

    # Get earnings yield
    earnings_yield = Fundamentals.earning_yield.latest

    # Get price to book ratio
    pb_ratio = Fundamentals.pb_ratio.latest

    inventory_turnover = Fundamentals.inventory_turnover.latest

    # Get return on invested capital
    roic = Fundamentals.roic.latest
    roic_5yr_avg = SimpleMovingAverage(inputs=[Fundamentals.roic], window_length=5 * 252)

    # Get f score
    f_score = Fscore(
        inputs=[net_income, roa, op_cash_flow, op_cash_flow - net_income, long_term_debt - long_term_debt_1yr_ago,
                current_ratio - current_ratio_1yr_ago, share_issued - share_issued_1yr_ago,
                gross_margin - gross_margin_1yr_ago, assets_turnover - assets_turnover_1yr_ago])

    # Get sector
    sector_code = Fundamentals.morningstar_sector_code.latest

    # Return Pipeline containing close_price and
    # sentiment_score that has our trading universe as screen
    return Pipeline(
        columns={
            'earnings_yield': earnings_yield,
            'pb_ratio': pb_ratio,
            'roic': roic,
            'roic_5yr_avg': roic_5yr_avg,
            'f_score': f_score,
            'sector_code': sector_code
        },
        screen=base_universe
    )


def before_trading_start(context, data):
    """
    Called every day before market open.
    """
    pass


def rebalance(context, data):
    log.info("Currently holding: " + ", ".join([position.symbol for position in context.portfolio.positions]))

    context.output = algo.pipeline_output('value_pipeline')
    context.sector_codes = {
        101.0: "Basic Materials",
        102.0: "Consumer Cyclical",
        103.0: "Financial Services",
        205.0: "Consumer Defensive",
        206.0: "Healthcare",
        207.0: "Utilites",
        308.0: "Communication Services",
        309.0: "Energy",
        310.0: "Industrials",
        311.0: "Technology"
    }

    sufficient_f_securities = context.output[context.output['f_score'] >= 7]

    for sector_code in context.sector_codes:
        df = sufficient_f_securities[sufficient_f_securities['sector_code'] == sector_code]
        df.reset_index(inplace=True)

        df['earnings_yield_rank'] = df.sort_values(by=['earnings_yield'], ascending=False).index
        df['pb_ratio_rank'] = df.sort_values(by=['pb_ratio'], ascending=True).index
        df['roic_rank'] = df.sort_values(by=['roic'], ascending=False).index
        df['roic_5yr_avg_rank'] = df.sort_values(by=['roic_5yr_avg'], ascending=False).index
        df['erp5_score'] = df['earnings_yield_rank'] + df['roic_rank'] + df['pb_ratio_rank'] + df['roic_5yr_avg_rank']
        df['erp5_rank'] = df.sort_values(by=['erp5_score']).index

        # Sell
        for stock in context.stocks_by_sector[sector_code]:
            if not df["index" == stock].any():  # F score below threshold
                order_target_percent(stock, 0)
                context.stocks_by_sector[sector_code].remove(stock)
            else:
                erp5_rank = df["index" == stock, "erp5_rank"]
                if erp5_rank > 15:
                    order_target_percent(stock, 0)
                    context.stocks_by_sector[sector_code].remove(stock)

        long_weight = 0.025  # TODO
        num_to_buy = 4 - len(context.stocks_by_sector[sector_code])
        if num_to_buy > 0:
            longs = df.sort_values(by="erp5_rank", ascending=True).head(num_to_buy)
            for stock in longs:
                context.stocks_by_sector[sector_code].add(stock)
                order_target_percent(stock, long_weight)

    # For each security in our universe, order long or short positions according
    # to our context.long_secs and context.short_secs lists.
    # for stock in context.security_list:
    #    if data.can_trade(stock) and context.sector_long_count[stock.]:
    #        order_target_percent(stock, long_weight)

    # Sell all previously held positions not in our new context.security_list.
    # for stock in context.portfolio.positions:
    #    if stock not in context.security_set and data.can_trade(stock):
    #        order_target_percent(stock, 0)

    # Log the long and short orders each week.
    log.info("This month's longs: " + ", ".join([long_.symbol for long_ in context.security_list]))


def record_vars(context, data):
    """
    Plot variables at the end of each day.
    """
    pass