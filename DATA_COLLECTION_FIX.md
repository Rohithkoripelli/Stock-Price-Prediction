# Fixed: Data Collection Now Includes T-1 (Yesterday)

## The Problem

When running on Jan 20, 2026, the system was only collecting data until Jan 16, 2026 instead of Jan 19, 2026 (T-1/yesterday).

**Expected:** Data until Jan 19 (yesterday)
**Actual:** Data until Jan 16 (4 days ago)

---

## Root Cause: yfinance End Date is EXCLUSIVE

### The Issue:

yfinance's `end` parameter is **EXCLUSIVE**, not inclusive.

```python
# OLD CODE (BROKEN):
END_DATE = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
# Result: END_DATE = '2026-01-19'
# yfinance fetches: 2019-01-01 to 2026-01-18 (excludes Jan 19!)
```

**Why it was confusing:**
- We calculated `END_DATE = yesterday`
- But yfinance excludes the end date
- So we got data until **day before yesterday**

### yfinance Behavior Test:

```python
import yfinance as yf

# Test 1: end='2026-01-19'
data1 = yf.Ticker('HDFCBANK.NS').history(start='2026-01-15', end='2026-01-19')
print(data1.index[-1])  # 2026-01-16 (excludes Jan 19!)

# Test 2: end='2026-01-20'
data2 = yf.Ticker('HDFCBANK.NS').history(start='2026-01-15', end='2026-01-20')
print(data2.index[-1])  # 2026-01-19 (includes Jan 19!)
```

**Conclusion:** To include Jan 19, we need `end='2026-01-20'` (today).

---

## The Fix

### Updated Code:

```python
# NEW CODE (FIXED):
END_DATE = datetime.now().strftime('%Y-%m-%d')
# Result: END_DATE = '2026-01-20' (today)
# yfinance fetches: 2019-01-01 to 2026-01-19 (includes yesterday!)
```

**Key insight:**
- To get data until yesterday (T-1), use today's date as the end parameter
- yfinance will exclude today and include yesterday

---

## Verification

### Before Fix:

```
$ python collect_stock_data.py
✓ Date Range: 2019-01-01 to 2026-01-16  ❌ (4 days old)
✓ Current Price: ₹931.10
```

### After Fix:

```
$ python collect_stock_data.py
✓ Date Range: 2019-01-01 to 2026-01-19  ✓ (yesterday!)
✓ Current Price: ₹927.90
```

**Result:** Fresh data until yesterday (T-1) as expected!

---

## Why Jan 16 → Jan 19 is a 3-Day Gap

Breaking down the days:

```
Jan 15 (Thu) - Market CLOSED (Maharashtra Elections)
Jan 16 (Fri) - Trading day ✓
Jan 17 (Sat) - Weekend
Jan 18 (Sun) - Weekend
Jan 19 (Mon) - Trading day ✓ ← Now included!
Jan 20 (Tue) - Today
```

**Last trading days:**
- Before fix: Jan 16 (Friday)
- After fix: Jan 19 (Monday)

---

## Impact on Daily Automation

### GitHub Actions Workflow:

The workflow runs at 10 PM IST daily. With this fix:

**Example: Workflow runs on Jan 20 at 10 PM:**
1. Collects data: 2019-01-01 to 2026-01-19 ✓
2. Current prices: As of Jan 19 close ✓
3. Predictions: For Jan 20 (next trading day) ✓

**Example: Workflow runs on Jan 21 at 10 PM:**
1. Collects data: 2019-01-01 to 2026-01-20 ✓
2. Current prices: As of Jan 20 close ✓
3. Predictions: For Jan 21 (next trading day) ✓

---

## Technical Details

### File Modified:
`collect_stock_data.py` (lines 33-38)

### Change Summary:
```diff
- END_DATE = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
+ END_DATE = datetime.now().strftime('%Y-%m-%d')
```

### Added Comment:
```python
# Note: yfinance end date is EXCLUSIVE, so we use today's date
# to include yesterday's data (T-1)
```

---

## Data Freshness Guarantee

After this fix, the system now guarantees:

**Daily at 10 PM IST:**
- Data collected until: T-1 (yesterday's closing prices)
- Technical indicators: Calculated on latest data
- Predictions: For T (next trading day)

**Example Timeline:**
```
10 PM IST Jan 20 → Collects until Jan 19 → Predicts Jan 20
10 PM IST Jan 21 → Collects until Jan 20 → Predicts Jan 21
10 PM IST Jan 22 → Collects until Jan 21 → Predicts Jan 22
```

**Note:** On weekends/holidays, last trading day is used automatically.

---

## Alternative Solutions Considered

### Option 1: Use `period='max'` (Rejected)
```python
data = yf.Ticker('HDFCBANK.NS').history(period='max')
```

**Why rejected:**
- Less control over start date
- Harder to debug issues
- Explicit date ranges are clearer

### Option 2: Add 1 day to yesterday (Overcomplicated)
```python
END_DATE = (datetime.now() - timedelta(days=1) + timedelta(days=1))
```

**Why rejected:**
- More confusing than just using `datetime.now()`
- Same result, more code

### Option 3: Use today's date (CHOSEN)
```python
END_DATE = datetime.now().strftime('%Y-%m-%d')
```

**Why chosen:**
- Simplest solution
- Clear intent with comment
- Works reliably with yfinance's exclusive end behavior

---

## Monitoring

### Check if data is fresh:

```bash
# Check collection summary
cat data/collection_summary.json | grep date_range

# Check HDFC Bank data
tail -1 data/stocks/private_banks/HDFCBANK_data.csv

# Check predictions
cat future_predictions_next_day.json | grep Current_Price
```

### Expected behavior:
- Date range should end at yesterday's date
- Current prices should match market close from yesterday
- Predictions should be for today

---

## Summary

**Problem:** yfinance `end` parameter is exclusive
**Solution:** Use today's date to include yesterday
**Result:** Fresh data until T-1 every day

**Date calculation:**
- **Old:** `END_DATE = yesterday` → Got data until day before yesterday
- **New:** `END_DATE = today` → Get data until yesterday

**This fix ensures predictions are always based on the most recent market data available!**
