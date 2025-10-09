# Fundamental Data Sources Documentation

This document outlines the **fundamental data sources** integrated into this project, including required environment variables, endpoints, and sample responses. It serves as a single reference for setting up, testing, and maintaining fundamental feature ingestion.

---

## 1. Alpha Vantage

- **Environment Variable**: `AV_API_KEY`
- **Endpoint**: https://www.alphavantage.co/query
- **Functions Used**:
  - `TIME_SERIES_DAILY_ADJUSTED` for historical daily prices
  - `OVERVIEW` for company fundamentals (e.g., `PE Ratio`, `EBITDA`)
- **Request Example**:
  ```bash
  curl "https://www.alphavantage.co/query?function=OVERVIEW&symbol=EURUSD&apikey=${AV_API_KEY}"
  ```
- **Expected Fields**:
  - `Symbol`
  - `AssetType`
  - `PERatio`
  - `EBITDA`
  - `DebtToEquity`
  - `EPS`
  - `BookValue`
  - `MarketCapitalization`
- **Sample Response**:
  ```json
  {
    "Symbol": "EURUSD",
    "AssetType": "Currency",
    "Name": "Euro / US Dollar",
    "PERatio": "15.23",
    "EBITDA": "N/A",
    "DebtToEquity": "0.00",
    "EPS": "0.00",
    "BookValue": "1.00",
    "MarketCapitalization": "N/A"
  }
  ```

---

## 2. Finnhub

- **Environment Variable**: `FINNHUB_API_KEY`
- **Endpoint**: https://finnhub.io/api/v1
- **Functions Used**:
  - `/stock/metric` for key metrics (e.g., `peBasicExclExtraTTM`, `ebitda`)
  - `/stock/financials-reported` for detailed financial statements
- **Request Example**:
  ```bash
  curl "https://finnhub.io/api/v1/stock/metric?symbol=EURUSD&metric=all&token=${FINNHUB_API_KEY}"
  ```
- **Expected Fields**:
  - `metric.peBasicExclExtraTTM`
  - `metric.ebitda`
  - `metric.debt/asset`
  - `metric.roeTTM`
- **Sample Response**:
  ```json
  {
    "metric": {
      "peBasicExclExtraTTM": 14.82,
      "ebitda": 2000000,
      "debt/asset": 0.12,
      "roeTTM": 0.08
    }
  }
  ```

---

## 3. Financial Modeling Prep (FMP)

- **Environment Variable**: `FMP_API_KEY`
- **Endpoint**: https://financialmodelingprep.com/api/v3
- **Functions Used**:
  - `/ratios/{symbol}` for financial ratios (e.g., `priceEarningsRatio`, `debtEquityRatio`)
  - `/income-statement/{symbol}` for income data (e.g., `ebitda`)
- **Request Example**:
  ```bash
  curl "https://financialmodelingprep.com/api/v3/ratios/EURUSD?apikey=${FMP_API_KEY}"
  ```
- **Expected Fields**:
  - `priceEarningsRatio`
  - `debtEquityRatio`
  - `ebitda`
  - `returnOnEquity`
- **Sample Response**:
  ```json
  [
    {
      "date": "2025-09-30",
      "symbol": "EURUSD",
      "priceEarningsRatio": 16.5,
      "debtEquityRatio": 0.10,
      "ebitda": 2500000,
      "returnOnEquity": 0.09
    }
  ]
  ```

---

## 4. API Ninja (Free)

- **Environment Variable**: Optional `API_NINJA_KEY` (no key required for basic usage)
- **Endpoint**: https://api.api-ninjas.com/v1/fact
- **Functions Used**:
  - `/currency` for currency-related facts or simple ratios
- **Request Example**:
  ```bash
  curl -H "X-Api-Key: ${API_NINJA_KEY}" \
       "https://api.api-ninjas.com/v1/currency?pair=EUR_USD"
  ```
- **Expected Fields**:
  - `pair`
  - `priceQuote`
  - `priceBase`
- **Sample Response**:
  ```json
  {
    "pair": "EUR_USD",
    "priceQuote": 1.0012,
    "priceBase": 0.9988
  }
  ```

---

## 5. Additional Sources

- **IEX Cloud** (optional): requires `IEX_API_KEY` – for advanced equity data
- **Twelve Data** (optional): requires `TWELVE_DATA_KEY` – broad market coverage

---

## 6. Data Storage & Processing

1. **Raw Data**: Store raw JSON responses under:
   ```
   data/raw/fundamentals/<source>/<symbol>_<YYYYMMDD>.json
   ```
2. **Processed Data**: Convert raw JSON to Parquet under:
   ```
   data/processed/fundamentals/<source>/<symbol>_<YYYYMMDD>.parquet
   ```
3. **Feature Loader**: The module `fundamentals.py` should read processed Parquet files and normalize feature names.

---

## 7. Testing & Validation

- **Schema Validation**: Ensure each fetch function returns a dictionary matching the required fields (see `tests/test_fundamental_features_schema.py`).
- **End-to-End Smoke Tests**: Use `scripts/test_fundamentals_endtoend.py` to fetch live data and validate persistence.

---

*Last updated: 2025-10-03*
