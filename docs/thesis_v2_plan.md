# Project Brief — HGE Gold Forecasting Thesis V2

## 1. تعریف پروژه

نسخه دوم یک سامانه پژوهشی و قابل بازتولید برای پیش‌بینی جهت حرکت قیمت طلا در افق‌های چندروزه است. این نسخه در کنار pipeline فعلی اجرا می‌شود و هیچ‌یک از خروجی‌های Legacy را حذف یا بازنویسی نمی‌کند.

## 2. مسئله

نسخه فعلی از مدل‌های محدود (`LogisticRegression`, `RandomForest` و baseline) و یک Hurst gate با وزن‌های ثابت استفاده می‌کند. بهترین نتیجه ثبت‌شده برای مسئله سه‌کلاسه جهت، حدود `Macro-F1 = 0.421` است. هدف V2 این است که:

1. مسئله اصلی پایان‌نامه را دقیق‌تر تعریف کند؛
2. نشت اطلاعات را با split زمانی Purged Walk-Forward کنترل کند؛
3. مدل‌های پایه متنوع و یک gate آموزش‌پذیر داشته باشد؛
4. معیار 60 درصد را با معیار ضدعدم‌توازن ثبت کند؛
5. خروجی‌های قابل دفاع، تست‌شده و قابل تکرار تولید کند.

## 3. کاربران هدف

- دانشجوی کارشناسی برای پایان‌نامه و دفاع؛
- استاد راهنما یا داور برای بازتولید نتایج؛
- توسعه‌دهنده برای ادامه پژوهش؛
- تحلیلگر کمی برای اجرای آزمایش روی داده معتبر.

## 4. محدوده

### داخل محدوده

- داده روزانه OHLCV طلا؛
- پیش‌بینی جهت قابل‌اقدام در افق‌های 1، 5، 10 و 20 روزه؛
- ویژگی‌های تکنیکال، نوسان، حجم، Hurst و regime؛
- مدل‌های خطی و ensemble در scikit-learn؛
- learned regime gate؛
- Locked Test، bootstrap و backtest پژوهشی؛
- خروجی CSV/JSON/Joblib؛
- تست واحد، یکپارچه و causality.

### خارج از محدوده فعلی

- ارسال سفارش واقعی؛
- اتصال مستقیم به MT5؛
- معاملات زنده؛
- داده خبری و NLP؛
- تضمین سود یا تضمین دقت 60 درصد روی بازار واقعی؛
- تنظیم پارامتر با استفاده از Locked Test.

## 5. تصمیم اصلی مسئله یادگیری

### تصمیم

مسئله اصلی از سه‌کلاسه `down/no_trade/up` به **جهت دودویی روی نمونه‌های actionable** تبدیل می‌شود. نمونه‌ای actionable است که قدر مطلق بازده آینده از threshold تطبیقی مبتنی بر نوسان و کف هزینه عبور کند.

### دلیل

کلاس `no_trade` در نسخه فعلی هم عدم‌توازن ایجاد می‌کند و هم مفهوم «تشخیص جهت» را مبهم می‌کند. مسئله دودویی actionable برای یک پایان‌نامه کارشناسی روشن‌تر، قابل‌اندازه‌گیری‌تر و از نظر مالی معنادارتر است.

### خروجی‌های ثانویه

برچسب سه‌کلاسه و بازده آینده همچنان در dataset نگه داشته می‌شوند تا خروجی‌های فعلی تا حد ممکن قابل مقایسه بمانند.

## 6. معیار پذیرش ثبت‌شده

معیار اصلی:

- `Balanced Accuracy >= 0.60`

قیود مکمل:

- `Macro-F1 >= 0.55`
- `Recall(up) >= 0.50`
- `Recall(down) >= 0.50`
- حداقل تعداد نمونه Locked Test طبق config
- انتخاب مدل بدون استفاده از Locked Test

عبور از این gate فقط برای همان dataset، بازه زمانی و protocol معتبر است و تضمین آینده بازار نیست.

## 7. معماری

```text
CSV / Sample Fixture
        |
        v
Strict OHLCV Validation
        |
        v
Causal Feature Engineering
        |
        +--> Feature Registry / Causality Test
        |
        v
Adaptive Actionable Labels
        |
        v
Development / Locked Test Split
        |
        v
Purged Walk-Forward Folds
        |
        +--> Logistic candidates
        +--> Random Forest
        +--> Extra Trees
        +--> HistGradientBoosting
        |
        v
OOF Probabilities
        |
        v
Learned Regime Gate / Best Base Fallback
        |
        v
Frozen Threshold + Final Refit
        |
        v
Locked Test Metrics + Bootstrap CI + Research Backtest
        |
        v
Manifest / Model Bundle / Compatibility Outputs
```

## 8. اجزای کد

| بخش | مسئولیت |
|---|---|
| `v2/config.py` | config typed و validation |
| `v2/data.py` | ingestion، schema و fixture |
| `v2/features.py` | feature engineering causal |
| `v2/targets.py` | target و threshold تطبیقی |
| `v2/splits.py` | Locked Test و Purged Walk-Forward |
| `v2/modeling.py` | candidateها، OOF و learned gate |
| `v2/evaluation.py` | metric، CI و backtest |
| `v2/pipeline.py` | orchestration و artifactها |
| `v2/cli.py` | رابط اجرای command line |

## 9. ویژگی‌ها

- lagged returns؛
- rolling mean/std؛
- momentum و drawdown؛
- slope و trend efficiency؛
- RSI، ATR و MACD؛
- skewness و kurtosis؛
- volume z-score و interaction؛
- rolling Hurst؛
- volatility/trend/Hurst regimes؛
- ویژگی‌های تقویمی cyclic.

Thresholdهای regime با rolling history و `shift(1)` ساخته می‌شوند تا از fit روی آینده یا کل نمونه جلوگیری شود.

## 10. مدل‌ها

- `LogisticRegression(C=0.2)`
- `LogisticRegression(C=1.0)`
- `RandomForestClassifier`
- `ExtraTreesClassifier`
- `HistGradientBoostingClassifier`

Gate از احتمال‌های OOF مدل‌های پایه و ویژگی‌های regime استفاده می‌کند. Gate فقط زمانی انتخاب می‌شود که روی آخرین fold توسعه از بهترین مدل پایه ضعیف‌تر نباشد؛ در غیر این صورت fallback به بهترین base انجام می‌شود.

## 11. خروجی‌های V2

خروجی‌های اصلی:

- `artifacts/v2/locked_test_metrics.csv`
- `artifacts/v2/candidate_selection_metrics.csv`
- `artifacts/v2/walk_forward_fold_metrics.csv`
- `artifacts/v2/backtest_summary.csv`
- `artifacts/v2/selected_model_map.json`
- `artifacts/v2/feature_registry.json`
- `artifacts/v2/execution_manifest.json`
- `data/predictions/v2/locked_test_predictions.csv`
- `models/v2/horizon_<H>_model_bundle.joblib`

خروجی‌های سازگار با نام‌گذاری قبلی:

- `artifacts/metadata/phase5_locked_test_metrics_report_v2.csv`
- `artifacts/metadata/phase4_selected_model_map_v2.json`
- `data/predictions/phase4/phase4_locked_test_predictions_v2.csv`

## 12. برنامه پیاده‌سازی و Milestoneها

### M1 — Baseline Preservation

- خروجی: شاخه مستقل و V2 بدون تغییر Legacy
- معیار پذیرش: اجرای تست‌های قدیمی بدون حذف فایل

### M2 — Data Contract

- خروجی: validator و schema داده واقعی
- وابستگی: CSV مرتب و بدون duplicate
- تست: invalid date، duplicate، OHLC violation، NaN

### M3 — Causal Features and Targets

- خروجی: feature matrix و target dataset
- تست: future mutation نباید feature گذشته را تغییر دهد

### M4 — Leakage-safe Validation

- خروجی: Locked Test و folds
- تست: `label_end_index < validation_start_row`

### M5 — Candidate Models and Gate

- خروجی: OOF probabilities و model bundle
- تست: selection بدون Locked Test و probability shape

### M6 — Evaluation

- خروجی: metrics، CI و backtest
- معیار پذیرش: gate ثبت‌شده، نه صرفاً Accuracy

### M7 — Real-data Experiment

- خروجی: اجرای نهایی با dataset معتبر، frozen config و hash
- معیار پذیرش: مشخص بودن source، بازه، adjustment و timestamp

### M8 — Thesis Documentation

- خروجی: معماری، روش تحقیق، سناریوهای تست، نتایج و محدودیت‌ها

## 13. تست‌ها

- Unit: data validation، feature functions، target alignment؛
- Causality: mutation داده آینده؛
- Split: purge و Locked Test boundary؛
- Modeling: OOF و عدم استفاده از test؛
- Integration: اجرای end-to-end و artifactها؛
- Statistical: bootstrap confidence interval؛
- Regression: عدم overwrite خروجی‌های Legacy.

## 14. ریسک‌ها

| ریسک | شدت | کنترل |
|---|---:|---|
| نرسیدن داده واقعی به 60% | زیاد | گزارش صادقانه، تغییر horizon/target فقط روی development |
| overfitting | زیاد | Locked Test، OOF، مدل‌های محدود و config frozen |
| class imbalance | متوسط | Balanced Accuracy، class weights و recall دو کلاس |
| regime instability | متوسط | learned gate با fallback |
| data leakage | زیاد | causal test، purge و manifest |
| داده continuous futures نامعتبر | زیاد | ثبت vendor، roll و adjustment قبل از claim |
| backtest خوش‌بینانه | زیاد | هزینه، نمونه‌های non-overlapping و محدودیت صریح |

## 15. Definition of Done

پروژه زمانی برای دفاع آماده است که:

1. dataset واقعی و provenance آن ثبت شده باشد؛
2. config آزمایش قبل از Locked Test freeze شده باشد؛
3. تمام تست‌ها Pass باشند؛
4. Locked Test فقط یک بار برای نتیجه نهایی استفاده شود؛
5. metrics، confusion matrix و CI ثبت شوند؛
6. نتیجه 60 درصد در صورت تحقق با baseline و CI گزارش شود؛
7. در صورت عدم تحقق، نتیجه منفی بدون دستکاری target گزارش شود؛
8. راهنمای نصب، اجرا و بازتولید موجود باشد؛
9. هیچ ادعای سودآوری بدون شواهد backtest معتبر مطرح نشود.
