# Project Enhancement & Robust Training Plan

This document outlines the plan to improve the model training pipeline based on our recent discussion.

## 1. The Problem

The current training script (`automated_training.py`) hangs and produces repetitive LightGBM warnings:
- `[LightGBM] [Warning] Stopped training because there are no more leaves that meet the split requirements`
- `[LightGBM] [Warning] No further splits with positive gain, best gain: -inf`

Stop all warnings and issues that prevent quality training missing data prevents quality training. 
See examples below:
- WARNING - Skipping Holloway for Weekly due to insufficient data (0 rows)
- WARNING - Unable to identify date column in data/EURUSD_H4.csv
Implement a way to resolve these issues and ensure these are not issues before we start training. 

The data is there we should be getting it. Now we need to write a unit test or something and make sure it passes when attempting to access data in the matter we desire it please. 

There is no reason to waste time if everything is not loaded. Remember the task? 



These issues stem from:
- **Data Insufficiency:** The model is being trained on datasets (especially for smaller timeframes like H4) that are too small for the default LightGBM parameters, leading to ineffective training.
- **Lack of Progress Indication:** The process appears to hang, with no clear feedback on progress or ETA.
- **No Timeout Mechanism:** Stalled training runs do not automatically terminate, requiring manual intervention.

## 2. The Solution: A Robust Training Pipeline

I will implement a multi-layered solution to make the training process more resilient, informative, and efficient.

### TODO List:

- [ ] **1. Create `robust_lightgbm_config.py`:** This new file will contain the core logic for a more intelligent training pipeline, including:
    -   Robust LightGBM parameter configurations for small and minimal datasets.
    -   A data quality diagnostic function to check for sufficient volume, variance, and class balance.
    -   An enhanced training function that uses these diagnostics to select the right parameters and handles errors gracefully.

- [ ] **2. Create `data_issue_fixes.py`:** This new utility will contain a `pre_training_data_fix` function. Its purpose is to perform an initial check on all data sources (`.csv` files) to ensure they contain a sufficient number of rows before any processing begins. This will prevent the "0 rows" errors seen previously.

- [ ] **3. Refactor `forecasting.py`:** I will modify the `HybridPriceForecastingEnsemble` class to use the new `enhanced_lightgbm_training_pipeline`. This will replace the old, less resilient training method.

- [ ] **4. Refactor `automated_training.py`:** I will update the main training script to:
    -   Call the `pre_training_data_fix` function at the very beginning.
    -   Integrate the changes from `forecasting.py`.
    -   Add a progress bar (`tqdm`) to the main training loop to provide clear feedback on which currency pair is being processed.

- [ ] **5. Implement Timeout/Hang Prevention:** I will add a custom LightGBM callback to the robust configuration. This callback will programmatically stop a training trial if it detects there is no improvement, preventing the script from getting stuck on a single trial.

## 3. Your Questions Answered

### Should we move the CSV data to a database (e.g., PostgreSQL)?

**Yes, this is an excellent idea for the next stage of development.**

*   **Efficiency & Performance:** A database like PostgreSQL is far more efficient for querying, filtering, and aggregating large datasets than reading CSV files into memory. Operations that are slow with Pandas on large files become nearly instantaneous.
*   **Data Integrity:** A database enforces a schema. This means no more "0 rows" errors or problems with inconsistent column names. Data is either valid and gets inserted, or it's rejected. This moves data validation to the point of data entry, which is much cleaner.
*   **Centralization & Scalability:** All data lives in one place, making it easier to manage, back up, and access from multiple applications (e.g., a future web backend, a separate analytics dashboard).

While I will focus on implementing the robust file-based training first, setting up a data ingestion pipeline into PostgreSQL is the logical next step to professionalize this system.

### Is not having a backend causing these hiccups?

**Not directly, but it's related.** The issues we're seeing (hanging, data errors) are happening in your *data processing and training pipeline*, which can be considered the "backend" of your machine learning system.

A traditional web backend (like the Django project in your workspace) is typically used for serving the *results* of the model via an API. While the Django backend isn't causing the training to fail, the lack of a robust data management system (like the PostgreSQL database we just discussed) is a major source of the problems.

By fixing the training pipeline and then migrating to a database, we are building a proper, robust backend for your entire trading signal system.

I will now begin executing the TODO list.