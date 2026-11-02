"""Compatibility entry point; fund valuation controls now live with fundamentals."""


def portfolio_valuations_page():
    from features.fundamental_attributes.page import fundamental_attributes_page

    fundamental_attributes_page()
