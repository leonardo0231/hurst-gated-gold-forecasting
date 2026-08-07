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

* داده روزانه OHLCV طلا؛
* پیش‌بینی دودویی جهت قیمت در افق‌های 1، 5، 10 و 20 روزه؛
* استفاده از تمام نمونه‌هایی که بازده آینده معتبر و پوشش ویژگی کافی دارند؛
* نگهداری برچسب ثانویه `down / flat / up` برای تحلیل شدت و جهت حرکت؛
* محاسبه وضعیت `actionable` به‌عنوان متغیر تشخیصی و تحلیلی، بدون استفاده از آن برای انتخاب نمونه‌های مدل؛
* ویژگی‌های تکنیکال، نوسان، حجم، Hurst و regime؛
* مدل‌های خطی و ensemble مبتنی بر scikit-learn؛
* learned regime gate با قابلیت fallback به بهترین مدل پایه؛
* Locked Test زمانی؛
* Purged Walk-Forward Validation؛
* Bootstrap Confidence Interval؛
* Backtest پژوهشی؛
* خروجی‌های CSV، JSON و Joblib؛
* تست‌های واحد، یکپارچه، causality، target alignment و split integrity.

### خارج از محدوده فعلی

* ارسال سفارش واقعی؛
* اتصال مستقیم Pipeline به MT5 برای معامله؛
* معاملات زنده؛
* داده خبری و NLP؛
* تضمین سودآوری؛
* تضمین دستیابی به دقت 60 درصد روی بازار واقعی؛
* تنظیم مدل یا threshold با استفاده از Locked Test؛
* انتخاب یا حذف نمونه‌های آموزشی بر اساس اطلاعاتی که فقط در آینده قابل مشاهده هستند.

## 5. تصمیم اصلی مسئله یادگیری

### تصمیم

مسئله اصلی V2 به‌صورت **Binary Direction Classification روی تمام نمونه‌های معتبر** تعریف می‌شود.

برای هر نمونه در زمان `t` و افق پیش‌بینی `H`، بازده لگاریتمی آینده محاسبه می‌شود:

```text
forward_log_return(t, H) = log(close[t + H]) - log(close[t])
```

Target اصلی به صورت زیر تعریف می‌شود:

```text
direction_binary = 1    if forward_log_return > 0
direction_binary = 0    if forward_log_return <= 0
```

نمونه فقط زمانی از Modeling Dataset کنار گذاشته می‌شود که Target آینده معتبر نباشد، پوشش ویژگی‌ها کافی نباشد، یا الزامات Data Contract و مرز زمانی را نقض کند.

مقدار حرکت آینده برای تصمیم‌گیری درباره ورود نمونه به Training، Validation یا Locked Test استفاده نمی‌شود.

شناسه فعلی این سیاست Target برابر است با:

```text
all_samples_binary_direction_v2_1
```

### دلیل

در نسخه قبلی V2، ابتدا با استفاده از مقدار بازده آینده مشخص می‌شد که یک نمونه `actionable` است یا خیر و فقط نمونه‌های actionable وارد مسئله Binary Classification می‌شدند.

این طراحی باعث ایجاد Future Selection Bias می‌شد، زیرا سیستم هنگام آموزش و ارزیابی عملاً از اطلاعاتی استفاده می‌کرد که در زمان تصمیم‌گیری واقعی در دسترس نبود: اینکه آیا در آینده حرکت قیمت به‌اندازه کافی بزرگ خواهد بود یا خیر.

در طراحی جدید، تمام نمونه‌های معتبر وارد مسئله پیش‌بینی جهت می‌شوند. در نتیجه مدل باید بدون اطلاع قبلی از بزرگی حرکت آینده، جهت قیمت را پیش‌بینی کند.

این تعریف به شرایط استفاده واقعی نزدیک‌تر است و امکان ارزیابی علمی معتبرتری از توان مدل فراهم می‌کند.

### برچسب‌های ثانویه

Threshold تطبیقی مبتنی بر نوسان همچنان محاسبه می‌شود:

```text
adaptive_threshold =
    threshold_k × rolling_volatility × sqrt(horizon)
```

همچنین یک کف ثابت برحسب basis point اعمال می‌شود.

این Threshold برای تعریف دو خروجی ثانویه استفاده می‌شود:

```text
direction_three_class:
    -1 = DOWN
     0 = FLAT
     1 = UP
```

و:

```text
is_actionable:
    True  = absolute future movement exceeds the adaptive threshold
    False = movement does not exceed the threshold
```

این اطلاعات برای تحلیل، مقایسه و ارزیابی مالی نگه داشته می‌شوند، اما `is_actionable` در تعیین `is_modeling_eligible` دخالت ندارد.

### خروجی‌های اصلی و ثانویه

خروجی اصلی Modeling:

```text
direction_binary
```

خروجی‌های تحلیلی ثانویه:

```text
forward_log_return
direction_three_class
direction_threshold
direction_threshold_bps
is_actionable
```

این جداسازی اجازه می‌دهد عملکرد مدل جهت مستقل از شدت حرکت آینده ارزیابی شود و در عین حال اطلاعات اقتصادی لازم برای تحلیل‌های بعدی حفظ شود.

## 6. معیار پذیرش ثبت‌شده

معیار اصلی ارزیابی:

* `Balanced Accuracy >= 0.60`

قیود مکمل:

* `Macro-F1 >= 0.55`
* `Recall(up) >= 0.50`
* `Recall(down) >= 0.50`
* حداقل تعداد نمونه Locked Test طبق Configuration؛
* انتخاب Model بدون استفاده از Locked Test؛
* انتخاب Probability Threshold بدون استفاده از Locked Test.

استفاده از Balanced Accuracy در کنار Macro-F1 و Recall دو کلاس باعث می‌شود عبور از معیار پذیرش صرفاً از طریق عدم‌توازن کلاس‌ها یا پیش‌بینی بیش‌ازحد یک جهت امکان‌پذیر نباشد.

عبور از این Acceptance Gate تنها برای Dataset، بازه زمانی، Feature Set، Target Policy و Validation Protocol همان آزمایش معتبر است و تضمینی درباره عملکرد آینده بازار ایجاد نمی‌کند.

نتایج Synthetic Research Fixture فقط برای اثبات رفتار صحیح نرم‌افزار و قابلیت یادگیری Pipeline استفاده می‌شوند و نباید به‌عنوان شواهد بازار واقعی گزارش شوند.

## 7. معماری

```text
Real CSV / Synthetic Research Fixture
                 |
                 v
        Strict OHLCV Validation
                 |
                 v
        Causal Feature Engineering
                 |
                 +----> Feature Registry
                 |
                 +----> Causality Tests
                 |
                 v
       Multi-Horizon Target Builder
                 |
                 +----> Binary Direction Target
                 |       (Primary Modeling Target)
                 |
                 +----> Adaptive Threshold
                 |
                 +----> DOWN / FLAT / UP
                 |       (Secondary Analysis)
                 |
                 +----> is_actionable
                         (Diagnostic Only)
                 |
                 v
      Modeling Eligibility Validation
       [NO future-actionability filter]
                 |
                 v
     Development / Locked Test Split
                 |
                 v
       Purged Walk-Forward Folds
                 |
        +--------+--------+--------+
        |        |        |        |
        v        v        v        v
     Logistic   RF    ExtraTrees   HGB
        |        |        |        |
        +--------+--------+--------+
                 |
                 v
          OOF Probabilities
                 |
                 v
     Learned Regime Gate Evaluation
                 |
                 +----> Gate selected
                 |
                 +----> Best Base fallback
                 |
                 v
      Frozen Probability Threshold
                 |
                 v
          Final Development Refit
                 |
                 v
           Locked Test Evaluation
                 |
        +--------+---------+
        |                  |
        v                  v
 Classification      Research Backtest
 Metrics + CI
        |                  |
        +--------+---------+
                 |
                 v
     Manifest / Predictions / Models
```

### اصل جلوگیری از Leakage

ویژگی‌های زمان `t` فقط می‌توانند از اطلاعات قابل دسترس تا همان زمان استفاده کنند.

Target برای آموزش طبیعتاً از آینده ساخته می‌شود، اما هیچ اطلاعات Target یا وضعیت آینده اجازه ورود به Feature Matrix یا فرآیند انتخاب نمونه در زمان پیش‌بینی را ندارد.

Training Foldهایی که Label آنها وارد محدوده Validation می‌شود با Purging حذف می‌شوند:

```text
training.label_end_index < validation_start_row
```

به همین ترتیب، Development Dataset نباید دارای Labelهایی باشد که وارد محدوده Locked Test می‌شوند.

Locked Test فقط پس از پایان فرآیند Model Selection و Threshold Selection برای ارزیابی نهایی استفاده می‌شود.


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
