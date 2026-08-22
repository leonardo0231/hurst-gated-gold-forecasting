# بررسی جهانی میزان یکتایی پروژه HGE Gold Forecasting

**تاریخ بررسی:** 2026-08-16  
**دامنه:** مقالات، preprintها، پایان‌نامه‌ها و مخازن عمومی جهانی درباره gold/XAUUSD forecasting، Hurst، regime-aware ML، multi-horizon forecasting، gating/Mixture-of-Experts و ارزیابی walk-forward.

## نتیجه اجرایی

در سطح جهانی نیز پروژه شما از نظر موضوع کلی «اولین» نیست. آثار متعددی وجود دارند که یکی یا چند جزء آن را پوشش می‌دهند:

- gold + Hurst + ML؛
- gold + regime-aware forecasting؛
- gold + multi-horizon forecasting؛
- gold + walk-forward؛
- financial ML + mixture-of-experts/gating؛
- financial ML + purged validation.

اما در جست‌وجوی هدفمند فعلی، مقاله یا مخزن معتبری پیدا نشد که **کل ترکیب** زیر را پیش‌تر با هم ارائه کرده باشد:

```text
daily XAUUSD OHLCV
+ binary directional targets at 1/5/10/20 days
+ causal rolling R/S Hurst-regime features
+ OOF probability-based learned gate over classical base models
+ best-base fallback
+ purged walk-forward validation
+ locked chronological test
+ threshold/gate selection isolated from locked test
+ block-bootstrap CI and provenance/leakage audits
```

بنابراین نتیجه دقیق جهانی چنین است:

> پروژه در سطح جهانی «اولین پروژه پیش‌بینی طلا با ML» نیست؛ اما در جست‌وجوی فعلی، ترکیب مشخص Hurst-regime + learned gate + multi-horizon directional XAUUSD + leakage-safe audit protocol یک ترکیب متمایز و بالقوه جدید است.

## 1. نزدیک‌ترین مقالات جهانی به موضوع طلا

### 1.1 Yang, Wang, Zeng & Li (2024) — نزدیک‌ترین اثر مستقیم

**عنوان:** *Improved prediction of global gold prices: An innovative Hurst-reconfiguration-based machine learning approach*  
**روش:** تحلیل multifractal، decomposition/reconfiguration مبتنی بر Hurst، embedding dimension و ensemble/optimization.  
**داده:** سه بازار عمده طلا در چین، آمریکا و بریتانیا.  
**نتیجه گزارش‌شده:** رابطه منفی Hurst و خطای پیش‌بینی و برتری مدل hybrid در خطای پیش‌بینی و direction accuracy نسبت به مدل‌های متعارف.  
**شباهت:** بسیار زیاد در gold + Hurst + ML.  
**تفاوت تعیین‌کننده:** Hurst در آن مقاله برای بازآرایی/تجزیه سری و انتخاب ساختار استفاده می‌شود؛ در HGE، Hurst یک feature/regime signal برای gate است. target، مدل‌های پایه، split و audit نیز متفاوت‌اند.  
**حکم:** نزدیک‌ترین precedent علمی، اما نه همان روش و نه همان پروژه. [منبع](https://doi.org/10.1016/j.resourpol.2023.104430)

### 1.2 Fikri (2025) — regime، macro، walk-forward و چندافقی

**عنوان:** *The Impact of Market Volatility Regimes on Gold Price Prediction Accuracy: A VIX-Based Machine Learning Approach*  
**روش:** regimeهای Calm/Normal/Crisis با VIX، ویژگی‌های VIX/DXY/S&P500، آزمون Granger، ARIMA/LSTM/GRU و walk-forward.  
**افق‌ها:** 1 و 7 روز.  
**نتیجه گزارش‌شده:** دقت جهت حدود 51.2%؛ عملکرد در بحران بدتر می‌شود؛ افق 1 روزه از 7 روزه بهتر است.  
**شباهت:** زیاد در gold + regime-aware + multi-horizon + walk-forward.  
**تفاوت:** regime بیرونی VIX است، نه Hurst؛ gate یادگیری‌پذیر بین expertها ندارد؛ target و مدل‌ها متفاوت‌اند.  
**اهمیت برای ادعای یکتایی:** این مقاله نشان می‌دهد بخش «regime-aware gold forecasting» دیگر جدید مطلق نیست، بنابراین نوآوری HGE باید روی Hurst-gated classical ensemble و ارزیابی audit-first متمرکز شود. [منبع](https://jurnal.uindatokarama.ac.id/index.php/djit/article/view/4555)

### 1.3 Iqbal & Eid (2025) — multi-horizon gold

**عنوان:** *Multi-Horizon Gold Price Forecasting and Its Implications for Financial Markets*  
**روش:** چارچوب deep learning و metaheuristic optimization برای چند بازه زمانی.  
**شباهت:** زیاد در multi-horizon gold forecasting و توجه به non-stationarity/regime dependence.  
**تفاوت:** تمرکز اصلی price forecasting و deep/metaheuristic models است، نه binary direction، Hurst-regime یا purged locked evaluation. [منبع](https://doi.org/10.54216/JSDGT.050204)

### 1.4 Dynamic algorithm selection برای سری‌های اقتصادی و طلا

چارچوب DYALP، dynamic segmentation و انتخاب مدل بر اساس regime را برای چند سری اقتصادی، از جمله قیمت روزانه طلا، پیشنهاد می‌کند و مدل‌هایی مثل LSTM، CNN-LSTM، XGBoost و SVR را با وزن‌دهی/انتخاب وابسته به regime ترکیب می‌کند.  
**شباهت:** بسیار زیاد در ایده dynamic model selection و regime-dependent experts.  
**تفاوت:** regime بر اساس similarity/segmentation تعریف می‌شود، نه Hurst؛ خروجی اصلی price error است؛ target direction چندافقی و pipeline کلاسیک HGE را ندارد. [منبع](https://ouci.dntb.gov.ua/en/works/lmjBPZO9/)

### 1.5 CNN-QRLSTM با sentiment و Hurst index (2026)

کارهای جدیدتر gold forecasting از CNN-QRLSTM، online news sentiment، EEMD و Hurst index استفاده می‌کنند. این نشان می‌دهد Hurst در ادبیات جهانی طلا در حال ترکیب‌شدن با decomposition و deep learning است.  
**تفاوت با HGE:** Hurst برای کنترل embedding/decomposition استفاده می‌شود، نه routing بین Logistic/RF/ExtraTrees/HGB؛ همچنین پروتکل locked test و purge مشابه پروژه حاضر در رکورد بررسی‌شده مشاهده نشد. [منبع](https://www.mdpi.com/1099-4300/28/3/271)

### 1.6 مدل‌های multifractal و Markov regime برای فلزات

مدل‌های precious-metal returns با fractional jump-diffusion، Markov regime-switching stochastic volatility و Hurst coefficient نیز وجود دارند. این آثار نشان می‌دهند Hurst می‌تواند شاخصی از persistence باشد، اما خود مقاله‌ها تصریح می‌کنند Hurst «مستقیماً بازده آینده را پیش‌بینی نمی‌کند» و بیشتر ویژگی likeness/persistence را توصیف می‌کند.  
**اهمیت:** از ادعای اینکه Hurst به‌تنهایی باید دقت 60% بسازد جلوگیری می‌کند. [منبع](https://www.mdpi.com/2227-7390/9/4/407)

## 2. نزدیک‌ترین آثار جهانی به gating و Mixture-of-Experts

### 2.1 Hybrid Recurrent Expert Gating (2026)

این مقاله RNN، LSTM و GRU را با softmax gating روی OHLCV سهام Google ترکیب می‌کند و گزارش می‌دهد gate در شرایط مختلف به expertهای متفاوت وزن می‌دهد.  
**شباهت:** زیاد در learned gate، expert specialization و non-stationarity.  
**تفاوت:** سهام Google، price regression، deep recurrent experts؛ فاقد Hurst، XAUUSD و پروتکل HGE. [منبع](https://doi.org/10.1016/j.procs.2026.06.366)

### 2.2 Adaptive Mixture-of-Experts در سری‌های مالی

کارهای MoE در پیش‌بینی سهام و سری‌های مالی، routing پویا، expert specialization و regime-dependent weighting را بررسی کرده‌اند. پروژه HGE از همین خانواده ایده می‌گیرد، اما gate آن shallow و قابل‌ممیزی است و expertهایش مدل‌های کلاسیک scikit-learn هستند.

### 2.3 Multi-Gate Mixture-of-Experts برای سبد و momentum

کارهای portfolio construction از multi-gate MoE برای وزن‌دهی چند تایم‌فریم و regimeهای مختلف استفاده کرده‌اند. این آثار به gate و multi-timeframe نزدیک‌اند، اما دارایی هدف، مسئله portfolio allocation و معماری deep آن‌ها با پیش‌بینی جهت تک‌دارایی طلا متفاوت است. [نمونه](https://www.dorienhermans.com/sites/default/files/Joel_Ong_Thesis.pdf)

### 2.4 نمونه جدید GoldSSM

پروژه/مقاله GoldSSM برای intraday gold direction forecasting، variable selection، چند مقیاس زمانی، regime context و stream gating را ترکیب می‌کند. این از نظر «gold + direction + gating + multiple temporal scales» به HGE نزدیک است.  
**تفاوت:** معماری state-space/deep، intraday، و فاقد شواهدی از rolling R/S Hurst، classical OOF gate، purged locked test و audit registry پروژه حاضر. [منبع](https://rahulsp.com/papers/goldssm)

## 3. نزدیک‌ترین آثار جهانی به validation و leakage control

### López de Prado و purged/CPCV

Purging، embargo، CPCV، PBO، PSR و DSR در ادبیات ML مالی جاافتاده‌اند و در پروژه‌های open-source نیز پیاده‌سازی شده‌اند. بنابراین purged validation به‌تنهایی نوآوری پروژه نیست؛ ارزش HGE در این است که آن را با gold/Hurst/gate و artifact audit یک‌جا اجرا می‌کند. [پیاده‌سازی open-source](https://github.com/eslazarev/purged-cross-validation)

### مقایسه با پروژه HGE

پروژه HGE در نسخه محلی از purge بر اساس `label_end_index`، locked test، عدم استفاده از test در انتخاب و moving-block bootstrap استفاده می‌کند. بااین‌حال، نسبت به برخی چارچوب‌های جهانی هنوز CPCV، PBO، deflated Sharpe و embargo قابل‌تنظیم کامل را ندارد. بنابراین باید گفت «پروتکل leakage-aware متمایز و مناسب thesis» نه «اولین پروتکل جهانی ضد leakage».

## 4. ماتریس شباهت جهانی

| اثر/چارچوب | Gold | Hurst | Regime | Gate/MoE | Multi-horizon | Direction | Purged/locked | شباهت کلی |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Yang et al. 2024 | ✓✓ | ✓✓ | غیرمستقیم | ensemble | محدود | ✓ | نامشخص | بسیار زیاد در Hurst-gold |
| Fikri 2025 | ✓✓ | — | ✓✓ VIX | — | ✓ | ✓ | walk-forward | زیاد |
| Iqbal & Eid 2025 | ✓✓ | — | ✓ | metaheuristic | ✓✓ | نامشخص | نامشخص | زیاد در multi-horizon |
| DYALP | ✓ | — | ✓✓ | dynamic selection | چندسری | price | نامشخص | زیاد در regime selection |
| CNN-QRLSTM 2026 | ✓✓ | ✓ | decomposition | — | نامشخص | price | نامشخص | زیاد در gold/Hurst |
| Hybrid Recurrent Expert Gating 2026 | — | — | ✓ | ✓✓ | نامشخص | price | نامشخص | زیاد در gate |
| GoldSSM | ✓✓ | — | ✓ | ✓✓ | multi-scale | ✓ | نامشخص | زیاد در gold/direction/gate |
| purgedcv | — | — | — | — | — | — | ✓✓ | زیاد در validation |
| HGE محلی | ✓✓ | ✓ | ✓ | ✓ | ✓✓ | ✓✓ | ✓✓ | ترکیب متمایز |

## 5. چه چیزهایی دیگر در جهان جدید نیستند؟

- پیش‌بینی قیمت طلا با شبکه عصبی، LSTM، CNN، GRU یا مدل hybrid؛
- استفاده از macro variables، VIX، DXY، S&P500، اخبار و Google Trends؛
- multi-horizon gold forecasting؛
- regime-aware gold prediction؛
- Hurst/multifractal analysis در طلا؛
- dynamic model selection و Mixture-of-Experts در سری‌های مالی؛
- walk-forward validation؛
- purged/embargo validation در financial ML.

## 6. چه چیزی هنوز در جست‌وجوی جهانی ترکیب متمایز پروژه است؟

1. تعریف دقیق target به‌صورت **binary direction از forward log-return** برای چند افق ثابت 1/5/10/20 روز.
2. استفاده از **rolling R/S Hurst در سطح feature و regime**، نه فقط تحلیل descriptive یا decomposition.
3. اتصال Hurst/trend/volatility regime به **gate یادگیری‌پذیر روی احتمال‌های OOF مدل‌های کلاسیک**.
4. وجود **best-base fallback** برای زمانی که gate در validation مزیت کافی نشان نمی‌دهد.
5. جداسازی صریح model selection، threshold selection و gate selection از locked test.
6. ترکیب این روش با feature registry، causal-feature tests، data-quality audit، SHA-256 provenance و artifact manifest.
7. تمرکز بر یک pipeline thesis-grade و offline، نه صرفاً notebook یا dashboard معاملاتی.

## 7. رتبه‌بندی ادعای یکتایی

| ادعا | ارزیابی جهانی |
|---|---|
| «اولین پروژه جهان برای پیش‌بینی طلا با ML» | نادرست. ادبیات بسیار گسترده است. |
| «اولین پروژه جهان برای استفاده از Hurst در پیش‌بینی طلا» | نادرست؛ Yang et al. و آثار multifractal پیش از آن وجود دارند. |
| «اولین مدل جهان برای multi-horizon gold forecasting» | نادرست؛ آثار متعدد این موضوع را پوشش داده‌اند. |
| «اولین استفاده از regime-aware ML برای gold» | قابل‌دفاع نیست؛ VIX-based و dynamic-regime کارهای مشابه وجود دارند. |
| «اولین learned gate در financial ML» | نادرست؛ MoE و expert gating سابقه دارند. |
| «ترکیب متمایز Hurst-regime + classical learned gate + multi-horizon XAUUSD direction + purged locked audit» | در جست‌وجوی فعلی، ادعایی قوی و بالقوه جدید است؛ برای ادعای قطعی نیاز به systematic review کامل دارد. |

## 8. نتیجه برای پایان‌نامه

بهترین framing جهانی برای پروژه این است:

> «این پژوهش یک الگوریتم بنیادی جدید در یادگیری ماشین معرفی نمی‌کند؛ contribution آن طراحی و ارزیابی یک pipeline ترکیبی، leakage-aware و قابل‌بازسازی برای پیش‌بینی جهت چندافقی XAUUSD است که Hurst-based regime features را با یک learned gate روی مدل‌های پایه ترکیب می‌کند.»

این framing هم با ادبیات جهانی سازگار است و هم جلوی ادعای اغراق‌آمیز را می‌گیرد. نوآوری شما بیشتر در **ترکیب، طراحی آزمایش، کنترل leakage و reproducibility** است، نه در ابداع Hurst، gate، Random Forest یا walk-forward به‌صورت جداگانه.

## 9. محدودیت جست‌وجو

این بررسی، جست‌وجوی هدفمند در موتور وب، ناشرها، arXiv، GitHub و چند منبع کد/پایان‌نامه است؛ ادعای crawl کامل Scopus، Web of Science، Google Scholar و همه رکوردهای paywalled را ندارد. همچنین بخشی از آثار جدید 2025–2026 ممکن است هنوز citation پایدار یا نمایه‌سازی کامل نداشته باشند. عبارت «یونیک» در این گزارش به معنی **unique combination in the searched evidence** است، نه اثبات ریاضی یا حقوقیِ اول‌بودن.
