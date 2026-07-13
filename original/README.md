# The original project — preserved on purpose

This folder contains the first version of this project, exactly as it was published, along with the
README that reported its results.

**It is kept here deliberately.** It is not an embarrassment to be hidden — it is the *subject* of
the analysis. [`nb/01_leakage_forensics.ipynb`](../nb/01_leakage_forensics.ipynb) quotes this code
line by line, reproduces its results, and then shows why they are not real. Deleting it would leave
that notebook arguing against something the reader cannot inspect.

## What is in here

| file | what it is |
|---|---|
| `HybridModel_Wavelet.ipynb` | the original ARFIMA + Wavelet + LSTM pipeline |
| `README_original.md` | the original write-up, reporting **R² = 0.83357** and **89.74% directional accuracy** on daily S&P 500 log returns |
| `hybrid_model_predictions.png` | the original return-forecast chart |
| `lstm_stock_price_prediction.png` | the original price-forecast chart |

## What was wrong with it

The headline numbers are an artifact of **data leakage**. In short:

```python
def combine_predictions(self, approx_pred, detail_preds, y_true):
    X = np.column_stack([approx_pred] + detail_preds)
    self.rf_model.fit(X, y_true)      # fits on the test set's true values…
    return self.rf_model.predict(X)   # …then "predicts" the same rows
```

`predict()` calls this with the **test set's own targets**. The model is shown the answers, memorises
them with a 100-tree forest, and returns them as a forecast.

The proof that this — and *only* this — produces the reported R²: replacing every LSTM output with
**pure random noise** reproduces the same R² ≈ 0.82. The ARFIMA, the wavelet and all five LSTMs
contribute nothing to the number.

Three further defects compounded it:

1. **Only 40 of 705 test days were ever evaluated** (`pywt.wavedec` downsamples; the pipeline
   truncated to the shortest coefficient array).
2. **Inputs came from the target's future** — the wavelet *coefficient index* was used as if it were
   a *day index*, feeding the model data from up to 735 days after the day it was predicting.
3. **There was no ARFIMA.** `auto_arima` selected ARIMA(2,0,0) — `d = 0` — and no fractional
   differencing appears anywhere in the code. It was an AR(2). Correctly so: on returns the
   fractional differencing parameter is ≈ 0, so there was no long memory to capture. That realisation
   is what redirected the whole project toward volatility.

**Do not cite the numbers in `README_original.md`.** They are reproduced in Part 1 solely to be
dismantled. The corrected results are in the [main README](../README.md).
