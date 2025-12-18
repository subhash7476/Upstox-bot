def select_best_option(chain, spot, direction):
    side = "CE" if direction.upper() == "BUY" else "PE"
    contracts = chain.get(side, [])

    def score(c):
        spread = (c.get("ask", 0) - c.get("bid", 0)) or 9999
        dist = abs(c["strike"] - spot)
        oi = c.get("oi", 0)
        return spread * 0.5 + dist * 0.01 - oi * 0.0001

    ranked = sorted(contracts, key=score)
    return ranked[:5]
