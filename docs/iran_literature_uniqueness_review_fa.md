# بررسی مقالات با وابستگی ایرانی و میزان یکتایی پروژه HGE

**تاریخ بررسی:** 2026-08-16  
**موضوع:** مقالات و پایان‌نامه‌های مرتبط با طلا، پیش‌بینی مالی، یادگیری ماشین، Hurst و رژیم‌های بازار با نویسنده یا وابستگی دانشگاهی ایرانی.

## نتیجه کوتاه

پروژه شما در ایران از نظر **ترکیب روش‌شناختی** متمایز است، اما از نظر «پیش‌بینی قیمت طلا با هوش مصنوعی» کاملاً بی‌سابقه نیست. در ایران سابقه قابل‌توجهی برای این موارد وجود دارد:

- پیش‌بینی قیمت طلا یا سکه با شبکه عصبی، neuro-fuzzy و GMDH؛
- استفاده از متغیرهای کلان مانند دلار، نفت، شاخص سهام و قیمت جهانی طلا؛
- استفاده از CNN، LSTM، Bi-LSTM و مدل‌های عمیق؛
- تحلیل Hurst و multifractality در شاخص بورس تهران؛
- مدل‌سازی regime switching برای بازده و نوسان بورس ایران.

اما در منابع ایرانیِ قابل‌یافتن، مقاله‌ای که هم‌زمان این مجموعه را داشته باشد پیدا نشد:

```text
XAUUSD daily OHLCV
        + binary direction at horizons 1/5/10/20
        + rolling R/S Hurst regimes
        + learned regime gate over base classifiers
        + purged walk-forward validation
        + locked chronological test
        + bootstrap CI, provenance, leakage/causality audit
```

بنابراین ادعای دقیق و قابل‌دفاع این نیست که «اولین پژوهش ایرانی در پیش‌بینی طلا با ML» هستید؛ ادعای مناسب‌تر این است:

> «یک چارچوب بازتولیدپذیر و leakage-aware برای پیش‌بینی جهت چندافقی XAUUSD با featureهای Hurst-regime و learned ensemble gate ارائه می‌شود؛ این ترکیب در ادبیات ایرانیِ بررسی‌شده یافت نشد.»

## 1. نزدیک‌ترین آثار ایرانی در پیش‌بینی طلا

### 1.1 سرفراز و افسر — neuro-fuzzy برای قیمت طلا

**عنوان فارسی:** بررسی عوامل مؤثر بر قیمت طلا و ارائه مدل پیش‌بینی قیمت آن به کمک شبکه‌های عصبی فازی  
**نویسندگان:** لیلا سرفراز، امیر افسر؛ در رکورد موجود، وابستگی دانشگاهی ایران برای نویسندگان ذکر شده است.  
**روش:** مدل neuro-fuzzy تاکاگی–سوگنو؛ مقایسه با رگرسیون.  
**موضوع:** قیمت طلای جهانی و عوامل اقتصادی، از جمله رابطه دلار و طلا.  
**نتیجه گزارش‌شده:** neuro-fuzzy از رگرسیون بهتر گزارش شده است.  
**شباهت با HGE:** متوسط در gold + nonlinear ML.  
**تفاوت:** price-level forecasting، بدون binary direction چندافقی، بدون Hurst، بدون gate و بدون purged/locked evaluation.

### 1.2 معمارنژاد و فرمان‌آرا (2011) — GMDH برای سکه طلای بورس کالا

**عنوان:** پیش‌بینی قیمت سکه طلا در بورس کالای ایران با رویکرد شبکه عصبی GMDH  
**نویسندگان:** عباس معمارنژاد، وحید فرمان‌آرا.  
**روش:** شبکه عصبی GMDH.  
**متغیرها:** نرخ دلار، قیمت سکه، قیمت طلای دلاری، نفت، شاخص کل سهام و تاریخ تحویل سکه.  
**نتیجه/نوآوری:** مدل‌سازی روابط غیرخطی و غربال متغیرهای مؤثر با GMDH.  
**شباهت با HGE:** زیاد در gold + nonlinear forecasting + استفاده از متغیرهای مالی.  
**تفاوت:** سکه آتی در بازار ایران، regression/price forecast، نه XAUUSD؛ Hurst و regime gate و اعتبارسنجی purged ندارد.

### 1.3 ناجی زواره (1391) — مدل ترکیبی برای قرارداد آتی سکه

**عنوان:** بررسی مقایسه‌ای بین مدل ترکیبی سیستم ژنتیک فازی ـ عصبی خودسازمانده و مدل خطی در پیش‌بینی قیمت توافقی قراردادهای آتی سکه طلا  
**نوع:** پایان‌نامه دانشگاه مازندران.  
**روش:** clustering با شبکه عصبی خودسازمانده، سپس سیستم ژنتیک فازی؛ مقایسه با مدل خطی.  
**شباهت:** زیاد در hybrid/ensemble و gold futures.  
**تفاوت:** regime/gating در معنای فعلی پروژه نیست؛ Hurst، چندافقی direction target و locked test گزارش نشده است.

### 1.4 اپرناک و قدسی (2013) — ANN برای سری زمانی طلا

**عنوان:** پیش‌بینی سری‌های زمانی با استفاده از شبکه عصبی مصنوعی در تعیین قیمت طلا  
**نویسندگان:** آرش اپرناک، رضا قدسی؛ اپرناک با دانشگاه تهران معرفی شده است.  
**روش:** شبکه عصبی مصنوعی برای پیش‌بینی قیمت طلا.  
**شباهت:** متوسط در gold time-series + ANN.  
**تفاوت:** تمرکز بر قیمت، نه جهت؛ فاقد Hurst، regime، purging و گزارش audit مدرن.

### 1.5 Amini & Kalantari (2024) — CNN-Bi-LSTM

**عنوان:** *Gold price prediction by a CNN-Bi-LSTM model along with automatic parameter tuning*  
**نویسندگان:** Amirhossein Amini، Robab Kalantari؛ هر دو از Khatam University تهران.  
**روش:** معماری‌های CNN/LSTM/Bi-LSTM با automatic parameter tuning.  
**داده:** closing gold prices در یک بازه 44 ساله، 1978 تا 2021.  
**نتیجه:** بهبود معیارهای پیش‌بینی price-level با ترکیب CNN و Bi-LSTM گزارش شده است.  
**شباهت:** زیاد در gold forecasting و temporal ML.  
**تفاوت:** deep regression و single-series close؛ no Hurst gate، no multi-horizon binary direction، no purged walk-forward و no locked-test governance.

### 1.6 Kianpoor، Fattahi و Hajian (2024) — متن، Google Trends و CNN

**عنوان:** *Gold Price Forecasting: A Novel Approach Based on Text Mining and Big-Data-Driven Model*  
**نویسندگان:** Saeed Kianpoor از Payame Noor University تهران، Shahram Fattahi از Razi University کرمانشاه، Mohsen Hajian از Payame Noor University.  
**روش:** Google Trends، استخراج متن اخبار، CNN و تقریباً 19,926 عنوان خبری.  
**نتیجه:** ترکیب اخبار و Google Trends برای پیش‌بینی قیمت طلا مؤثر گزارش شده است.  
**شباهت:** زیاد در gold + ML، و از نظر مسئله چندمنبعی به توسعه آتی HGE نزدیک است.  
**تفاوت:** news/text-driven، نه Hurst/technical regime؛ پروتکل leakage-safe پروژه حاضر گزارش نشده است.

### 1.7 Abdolalizadeh — مدل‌های SARIMAX، SVM و LSTM برای طلای ایران

**عنوان:** *Forecasting Gold Prices in Iran Using Machine Learning and Network Models: Assessing the Impact of Macroeconomic Factors*  
**نویسنده:** Muhammad Abdolalizadeh، فارغ‌التحصیل آمار اقتصادی و اجتماعی دانشگاه ولی‌عصر رفسنجان.  
**داده:** قیمت روزانه بازار ایران از TGJU، 2014 تا 2025؛ OHLC.  
**روش:** SARIMAX با و بدون متغیرهای کلان، SVM و LSTM برای پیش‌بینی روز بعد.  
**نتیجه گزارش‌شده:** افزودن دلار آمریکا، نفت برنت و شاخص بورس تهران بهبود معناداری ایجاد نکرده است.  
**شباهت:** زیاد در daily gold، مقایسه مدل و بررسی macro features.  
**تفاوت:** gold market ایران، next-day price forecasting؛ Hurst، چندافقی، gate و purged validation ندارد.

### 1.8 Tashakkori et al. (2024) — MLP برای طلا

**عنوان:** *Forecasting Gold Prices with MLP Neural Networks: A Machine Learning Approach*  
**نویسندگان:** Arash Tashakkori، Fatemeh Salboukh، Hossein Talebzadeh و همکاران؛ Talebzadeh از واحد علوم و تحقیقات دانشگاه آزاد تهران معرفی شده است.  
**روش:** MLP با قیمت‌های تاریخی و شاخص‌های اقتصادی.  
**نتیجه گزارش‌شده:** خطای تست نزدیک 0.001 در رکورد/چکیده مقاله ذکر شده است.  
**شباهت:** متوسط تا زیاد در gold + historical/economic features + ML.  
**تفاوت:** regression/price level، بدون Hurst و validation protocol هم‌سطح HGE.

## 2. آثار ایرانی نزدیک به Hurst و حافظه بلندمدت

### 2.1 Norouzzadeh & Jafari (2005) — multifractal TEPIX

**عنوان:** *Application of Multifractal Measures to Tehran Price Index*  
**نویسندگان:** P. Norouzzadeh، G. R. Jafari.  
**روش:** R/S، modified R/S به روش Lo، DFA و generalized Hurst exponents.  
**نتیجه:** بررسی long-memory، multifractality و ویژگی‌های scaling در TEPIX؛ Hurst بالاتر از 0.5 و long-term dependence در برخی تحلیل‌ها گزارش شده است.  
**شباهت:** بسیار زیاد در Hurst/R-S و financial time series.  
**تفاوت:** تحلیل بازار سهام ایران، نه ML forecasting روی طلا؛ gate و هدف جهت ندارد.

### 2.2 Moeini، Ahrari و Madarshahi (2007) — chaos و Hurst در بورس تهران

**عنوان:** *Investigating Chaos in Tehran Stock Exchange Index*  
**روش:** correlation dimension، Hurst exponent و largest Lyapunov exponent.  
**نتیجه:** شواهدی از ساختار غیرخطی/رفتار آشوبی در شاخص بورس تهران گزارش شده است.  
**شباهت:** زیاد در غیرخطی‌بودن و Hurst به‌عنوان descriptor بازار.  
**تفاوت:** تحلیل ساختار، نه مدل supervised جهت طلا.

### 2.3 مطالعات multifractal رفتار معامله‌گران تهران

مطالعاتی روی حجم معاملات معامله‌گران حقیقی و نهادی در بورس تهران، generalized Hurst exponents و multifractal dimensions را محاسبه کرده‌اند و نقش همبستگی بلندمدت و fat tails را بررسی کرده‌اند. این کارها نشان می‌دهند استفاده از Hurst در حوزه مالی ایران سابقه دارد، اما Hurst را به learned gate برای انتخاب مدل‌های پیش‌بینی XAUUSD وصل نمی‌کنند.

### 2.4 Fractal analysis و کارایی بازار ایران در دوره COVID-19

مقاله *Fractal analysis and the relationship between efficiency of capital market indices and COVID-19 in Iran* از modified Hurst exponent برای بررسی کارایی بازار سرمایه ایران و اثر COVID-19 استفاده می‌کند.  
**شباهت:** Hurst + regime/efficiency + بازار ایران.  
**تفاوت:** هدف، تحلیل کارایی و بحران است، نه prediction pipeline چندافقی با ML gate.

## 3. آثار ایرانی نزدیک به regime switching

### 3.1 Abtahi & Nikfetrat (2012)

**عنوان:** شناسایی چرخش رژیم در بازده بازار اوراق بهادار ایران  
**روش:** Markov regime switching روی شاخص قیمت و بازده نقدی تهران در دوره 1385 تا 1390.  
**نتیجه:** سه رژیم شناسایی شده؛ یک رژیم با میانگین بازده منفی و دو رژیم با میانگین مثبت.  
**شباهت:** زیاد در regime-dependent financial behavior.  
**تفاوت:** regime estimation آماری است، نه learned gate روی expertهای ML و نه gold.

### 3.2 Nazifi & Fatahi (2012)

**عنوان:** *Regime Switching GARCH Models and GARCH Models, in Stock Market of the Developing Countries*  
**روش:** مقایسه GARCH و Markov-switching GARCH برای نوسان و پیش‌بینی out-of-sample بازار تهران.  
**شباهت:** زیاد در regime، volatility و out-of-sample.  
**تفاوت:** GARCH/volatility، نه classification جهت طلا و نه Hurst gate.

### 3.3 مدل‌های جدید بورس ایران با deep learning و hybrid optimization

مقاله *Deep Learning for Stock Market Prediction* با وابستگی نویسندگان ایرانی، مدل‌های Decision Tree، Bagging، Random Forest، AdaBoost، Gradient Boosting، XGBoost، ANN، RNN و LSTM را برای گروه‌هایی از سهام ایران مقایسه می‌کند. همچنین کارهای جدیدتر hybrid RNN و metaheuristic برای شرکت‌های ایرانی یا داده‌های مالی مرتبط گزارش شده‌اند.  
**شباهت:** زیاد در مدل‌سازی ML مالی و مقایسه چند الگوریتم.  
**تفاوت:** stock selection/price prediction در بورس ایران، فاقد ترکیب مشخص Hurst + XAUUSD + purged gate.

## 4. ماتریس شباهت

امتیازها کیفی‌اند و به معنی مقایسه عددی performance نیستند.

| اثر ایرانی | طلا | ML | Hurst | Regime/gate | جهت چندافقی | purged/locked/audit | شباهت کلی |
|---|---:|---:|---:|---:|---:|---:|---|
| سرفراز و افسر | ✓ | ✓ | — | — | — | — | متوسط |
| معمارنژاد و فرمان‌آرا | ✓ | ✓ | — | — | — | — | زیاد |
| ناجی زواره | ✓ | ✓ | — | خوشه‌بندی | — | — | زیاد |
| اپرناک و قدسی | ✓ | ✓ | — | — | — | — | متوسط |
| Amini & Kalantari | ✓ | ✓✓ | — | — | — | — | زیاد |
| Kianpoor et al. | ✓ | ✓✓ | — | — | — | — | زیاد |
| Abdolalizadeh | ✓ | ✓ | — | — | next-day | نامشخص | زیاد |
| Tashakkori et al. | ✓ | ✓ | — | — | — | نامشخص | متوسط-زیاد |
| Norouzzadeh & Jafari | — | — | ✓✓ | — | — | — | زیاد در Hurst |
| Moeini et al. | — | — | ✓ | chaos | — | — | متوسط در Hurst |
| Abtahi & Nikfetrat | — | — | — | ✓✓ | — | out-of-sample | زیاد در regime |
| Nazifi & Fatahi | — | — | — | ✓✓ | — | out-of-sample | زیاد در regime |
| HGE محلی | ✓✓ | ✓ | ✓ | ✓ | ✓✓ | ✓✓ | ترکیب متمایز |

## 5. ارزیابی دقیق میزان یکتایی پروژه

### چیزهایی که در ایران جدید نیستند

- پیش‌بینی قیمت طلا با AI/ANN؛
- استفاده از قیمت طلا، دلار، نفت و شاخص سهام؛
- مدل‌های hybrid، fuzzy، GMDH، CNN، LSTM و MLP؛
- تحلیل Hurst و multifractal در بازار مالی ایران؛
- regime switching در بازده و نوسان بورس تهران؛
- مقایسه چند مدل ML.

### چیزهایی که در جست‌وجوی ایرانی پیدا نشدند

- پیش‌بینی **جهت** XAUUSD، نه صرفاً price level؛
- چهار horizon مشخص 1/5/10/20 روزه در یک pipeline؛
- استفاده از rolling Hurst به‌عنوان regime feature با کنترل causal؛
- learned logistic gate روی احتمال‌های OOF مدل‌های پایه؛
- fallback به بهترین base model؛
- purge بر اساس `label_end_index` برای جلوگیری از label overlap؛
- locked chronological test جدا از model/threshold/gate selection؛
- moving-block bootstrap CI و معیارهای balanced accuracy، macro-F1، recall دو کلاس و MCC؛
- provenance، SHA-256، feature registry، data audit و تست‌های causality در یک چارچوب بازتولیدپذیر.

## 6. درجه‌بندی ادعای نوآوری

| ادعا | ارزیابی |
|---|---|
| «اولین ایرانی که قیمت طلا را با یادگیری ماشین پیش‌بینی کرده» | نادرست/قابل‌دفاع نیست؛ آثار متعددی قبل از پروژه وجود دارند. |
| «اولین ایرانی که از Hurst در تحلیل مالی استفاده کرده» | نادرست؛ TEPIX و مطالعات multifractal سابقه دارند. |
| «اولین پروژه ایرانی با Hurst + gold + ML» | ممکن است، اما برای ادعای قطعی به مرور نظام‌مند کامل‌تر نیاز دارد. |
| «چارچوبی متمایز برای Hurst-regime + learned gate + multi-horizon XAUUSD + leakage-safe evaluation» | ادعای قوی و در جست‌وجوی فعلی پشتیبانی‌شده است. |
| «مدل برتر بازار با دقت 60%» | فعلاً پشتیبانی نمی‌شود؛ خروجی locked test پروژه معیار 60% را رد نکرده است. |

## 7. منابع ایرانی منتخب

- [Amini & Kalantari (2024), CNN-Bi-LSTM gold prediction](https://doi.org/10.1371/journal.pone.0298426)
- [Kianpoor, Fattahi & Hajian (2024), text mining and Google Trends](https://doi.org/10.2478/sbe-2024-0049)
- [Abdolalizadeh, Iranian gold forecasting with SARIMAX/SVM/LSTM](https://isnac.ir/AAHA-AAHDKF)
- [معمارنژاد و فرمان‌آرا، GMDH برای سکه طلا](https://sciexplore.ir/Documents/Details/988-716-026-122?Title=Anticipation+of+Iran+Mercantile+Exchange+%28IME%29+gold+coin+price+using+Artificial+Neural+Network+Approach+with+GMDH+Algorithm)
- [ناجی زواره، پایان‌نامه مدل ترکیبی قرارداد آتی سکه](https://www.virascience.com/thesis/577131/)
- [اپرناک و قدسی، ANN برای قیمت طلا](https://www.researchgate.net/publication/299863866_pysh_byny_sryhay_zmany_ba_astfadh_az_shbkh_sby_msnwy_dr_tyyn_qymt_tla_Forecasting_time_series_using_artificial_neural_networks_in_the_gold_price)
- [سرفراز و افسر، neuro-fuzzy gold forecasting](https://mpra.ub.uni-muenchen.de/2855/)
- [Norouzzadeh & Jafari (2005), multifractal TEPIX](https://doi.org/10.1016/j.physa.2005.02.046)
- [Moeini et al. (2007), chaos in Tehran Stock Exchange](https://ideas.repec.org/a/eut/journl/v12y2007i1p103.html)
- [Abtahi & Nikfetrat (2012), regime switching in Iranian stock returns](https://sciexplore.ir/Documents/Details/027-895-656-900?Title=Identifying+Regime+Switching+of+Stock+Market+Returns+in+Iran)
- [Nazifi & Fatahi, regime-switching GARCH for Tehran market](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1987428)
- [Fractal analysis and capital-market efficiency in Iran](https://doi.org/10.1016/j.rinp.2021.104262)
- [Deep Learning for Stock Market Prediction](https://arxiv.org/abs/2004.01497)

## 8. محدودیت و نتیجه نهایی

پایگاه‌های ایرانی مانند SID، Magiran، Civilica و IranDoc در جست‌وجوی عمومی همیشه index کامل یا متن قابل‌خزش ارائه نمی‌کنند؛ بسیاری از پایان‌نامه‌ها و مقاله‌ها فقط با جست‌وجوی فارسی، نام نویسنده یا عنوان دقیق پیدا می‌شوند. بنابراین این گزارش «مرور هدفمند قابل‌ردگیری» است، نه گواهی مطلق نبود هیچ اثر مشابه.

با این محدودیت، پروژه شما در ایران **از نظر موضوع کلی تکراری نیست، اما از نظر ترکیب روش‌ها متمایز و بالقوه یونیک است**. ارزش اصلی آن نه در استفاده از یک الگوریتم جدید، بلکه در کنار هم قراردادن Hurst-regime، learned gate، چندافقی‌بودن و پروتکل ارزیابی leakage-safe است. برای تثبیت ادعای نوآوری، پایان‌نامه باید دقیقاً همین مرز را بیان کند و از ادعای «پیش‌بینی دقیق» یا «اولین کار AI در طلا» پرهیز کند.
