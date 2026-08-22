# فهرست مشکلات پروژه بر اساس ادبیات هر بخش

## خلاصه مدیریتی

مشکلات پروژه به دو گروه تقسیم می‌شوند:

- **مشکلات بنیادی:** target پرنویز، نبود متغیرهای برون‌زای مؤثر بر طلا، non-stationarity، ضعف Hurst estimator و عدم تطابق معیار آماری با هدف اقتصادی.
- **کمبودهای روش‌شناختی:** تعداد محدود مدل‌ها، gate و threshold ناپایدار، نبود CPCV/PBO/DSR، provenance ناقص و اعتبارسنجی محدود.

عملکرد فعلی این تشخیص را تأیید می‌کند: balanced accuracy در locked test برای افق‌های 1/5/10/20 برابر `0.551/0.523/0.483/0.476` و AUC برابر `0.572/0.533/0.503/0.466` است.

## 1. بخش هدف‌گذاری و تعریف مسئله

### مشکل 1 — هدف binary sign بیش از حد پرنویز است

**در پروژه:** برچسب برابر sign بازده بسته‌شدن امروز تا بسته‌شدن H روز بعد است؛ حتی حرکت‌های بسیار کوچک و غیرقابل‌معامله نیز برچسب UP/DOWN می‌گیرند.

**ارتباط با ادبیات:** surveys مالی هشدار می‌دهند که تعریف direction بین مطالعات استاندارد نیست و مقایسه accuracy را دشوار می‌کند. کارهای gold معمولاً price-level، return، volatility یا multi-horizon error را بررسی می‌کنند و همه حرکت‌ها را به دو کلاس قطعی تبدیل نمی‌کنند.

**اثر:** label noise بالا می‌رود و مدل برای حرکت‌هایی آموزش می‌بیند که بعد از cost/slippage ارزش اقتصادی ندارند.

**شدت:** بحرانی.

### مشکل 2 — target آماری و backtest اقتصادی با هم هم‌راستا نیستند

**در پروژه:** مدل روی sign همه بازده‌ها آموزش می‌بیند، اما backtest برای هر معامله 5 bps هزینه کم می‌کند؛ `is_actionable` فقط diagnostic است و در target دخالت ندارد.

**اثر:** ممکن است مدل از نظر آماری یک جهت کوچک را درست پیش‌بینی کند، اما بعد از هزینه زیان‌ده باشد؛ یا برای رسیدن به 60% مجبور به یادگیری نویزی شود.

**شدت:** زیاد.

### مشکل 3 — چهار افق با یک feature set تقریباً مشترک مدل‌سازی می‌شوند

**در پروژه:** horizonهای 1، 5، 10 و 20 با featureهای اصلی 5 تا 126 روزه و یک خانواده مدل مشترک اجرا می‌شوند.

**اثر:** افق 20 به ساختارهای بلندمدت‌تر، متغیرهای کلان یا target متفاوت نیاز دارد؛ خروجی فعلی نشان می‌دهد signal با horizon از بین می‌رود.

**شاهد:** AUC افق 20 برابر `0.466` و recall UP فقط `0.040` است.

**شدت:** بحرانی.

## 2. بخش داده و بازار طلا

### مشکل 4 — داده فقط OHLCV داخلی طلاست

**در پروژه:** featureها مشتقات close/high/low/open/volume هستند؛ DXY، real yield، نرخ بهره، VIX، S&P500، نفت، نقره، ETF، positioning و news وجود ندارد.

**ارتباط با ادبیات:** پژوهش‌های جدید gold regime-aware از VIX، DXY و S&P500 استفاده می‌کنند و VIX change را predictor مهم معرفی می‌کنند؛ کارهای دیگری از news و Google Trends استفاده می‌کنند. [Fikri 2025](https://jurnal.uindatokarama.ac.id/index.php/djit/article/view/4555)، [Kianpoor et al.](https://doi.org/10.2478/sbe-2024-0049)

**اثر:** مدل باید اثر شوک‌های کلان را از price history استخراج کند؛ در بحران‌ها و regimeهای جدید این اطلاعات در OHLCV گذشته وجود ندارد.

**شدت:** بحرانی.

### مشکل 5 — provenance داده کامل نیست

**در پروژه:** `symbol=XAUUSD` و `timeframe=D1` ثبت شده، اما broker، server، timezone و export_date خالی هستند.

**اثر:** مشخص نیست کندل روزانه در تمام دوره‌ها چه session و چه ساعت UTC را پوشش می‌دهد؛ تغییر session می‌تواند هم feature و هم label را جابه‌جا کند.

**شدت:** زیاد.

### مشکل 6 — volume احتمالاً tick volume است و قابل‌تعمیم نیست

**در پروژه:** volume به‌عنوان feature وارد می‌شود، اما نوع آن MT5 broker volume است و broker مشخص نشده.

**اثر:** حجم broker-specific است و ممکن است در داده broker دیگر یا بازار spot متفاوت باشد.

**شدت:** متوسط تا زیاد.

### مشکل 7 — outlierها audit شده‌اند اما اثرشان مدل‌سازی نشده است

**شاهد:** audit شامل 9 flash move، 11 robust outlier و 171 ردیف suspicious است.

**اثر:** Hurst، rolling volatility، skew/kurtosis و threshold volatility به outlier حساس‌اند. «فیلترنکردن» برای جلوگیری از selection bias قابل‌دفاع است، اما باید sensitivity analysis گزارش شود.

**شدت:** زیاد.

## 3. بخش Hurst و regime

### مشکل 8 — Hurst estimator پروژه یک برآورد تک‌مقیاسی ساده است

**در پروژه:** `_hurst_rs` عملاً `log(R/S)/log(n)` را برای یک rolling window محاسبه می‌کند.

**ارتباط با ادبیات:** مطالعات TEPIX از R/S، modified R/S به روش Lo، DFA و generalized Hurst استفاده می‌کنند؛ کارهای gold نیز Hurst را در decomposition و multifractal framework قرار می‌دهند. [Norouzzadeh & Jafari](https://doi.org/10.1016/j.physa.2005.02.046)، [Yang et al.](https://doi.org/10.1016/j.resourpol.2023.104430)

**اثر:** estimator می‌تواند به finite sample، trend، heavy tails و nonstationarity حساس باشد و با H واقعی long memory اشتباه شود.

**شدت:** بحرانی برای contribution Hurst.

### مشکل 9 — Hurst به‌عنوان feature هست، اما incremental value آن اثبات نشده

**در پروژه:** Hurst وارد feature set و meta-gate شده، اما ablation رسمی `بدون Hurst / با Hurst` در artifacts وجود ندارد.

**اثر:** هنوز مشخص نیست Hurst واقعاً بهتر از momentum، volatility یا trend regime عمل می‌کند یا صرفاً پیچیدگی اضافه کرده است.

**شدت:** بحرانی برای ادعای نوآوری.

### مشکل 10 — Hurst persistence را توصیف می‌کند، نه جهت را

**ادبیات:** Qian & Rasheed و Eom et al. رابطه Hurst و predictability را گزارش می‌کنند، اما Hurst جهت آینده را مستقیماً تعیین نمی‌کند. مدل‌های precious-metal نیز Hurst را بیشتر descriptor persistence می‌دانند. [Qian & Rasheed](https://m.actapress.com/Abstract.aspx?paperId=17650)، [Eom et al.](https://arxiv.org/abs/0712.1624)، [precious metals](https://www.mdpi.com/2227-7390/9/4/407)

**اثر:** انتظار دقت 60% فقط به‌دلیل اضافه‌کردن Hurst از نظر علمی موجه نیست.

**شدت:** زیاد.

### مشکل 11 — regimeها بر اساس quantile هستند، نه regimeهای validated

**در پروژه:** `hurst_regime` با quantileهای 33%/67% و `trend_regime` با آستانه ثابت 0.45 ساخته می‌شود.

**اثر:** این regimeها descriptive هستند و معلوم نیست با تفاوت واقعی در conditional return یا model error هم‌راستا باشند.

**شدت:** زیاد.

## 4. بخش مدل و learned gate

### مشکل 12 — expertها واقعاً regime-specialized نیستند

**در پروژه:** Logistic, RF, ExtraTrees و HGB روی کل development آموزش می‌بینند و gate فقط probability و چند regime feature را ترکیب می‌کند.

**ارتباط با ادبیات:** MoEها معمولاً expertهایی دارند که در sub-regime یا subtask تخصص پیدا می‌کنند؛ dynamic model selection نیز مدل را به regime مشخص وصل می‌کند. [Hybrid Recurrent Expert Gating](https://doi.org/10.1016/j.procs.2026.06.366)، [DYALP](https://ouci.dntb.gov.ua/en/works/lmjBPZO9/)

**اثر:** gate انتخاب می‌کند، اما expertهای آن دانش تخصصی جداگانه ندارند؛ بنابراین ممکن است فقط probability noise را دوباره وزن‌دهی کند.

**شدت:** زیاد.

### مشکل 13 — gate و threshold روی یک fold محدود انتخاب می‌شوند

**در پروژه:** آخرین fold توسعه برای threshold و مقایسه gate با بهترین base استفاده می‌شود؛ سپس مدل نهایی روی توسعه refit می‌شود.

**اثر:** حدود 573 تا 583 نمونه برای انتخاب threshold و strategy در بازار nonstationary محدود است؛ افت validation به locked test این ناپایداری را نشان می‌دهد.

**شدت:** زیاد.

### مشکل 14 — calibration احتمال‌ها بررسی نشده است

**در پروژه:** gate از probabilityهای مدل‌های پایه استفاده می‌کند، اما calibration curve، reliability diagram، isotonic/Platt calibration یا calibration by regime وجود ندارد.

**اثر:** probabilityهای بدکالیبره ورودی نامناسبی برای gate و threshold هستند؛ confidence ممکن است با skill واقعی اشتباه شود.

**شدت:** متوسط تا زیاد.

### مشکل 15 — فضای مدل محدود است، اما مشکل اصلی احتمالاً این نیست

**در پروژه:** Logistic Regression، RF، ExtraTrees و HGB وجود دارند؛ SVM، XGBoost، LSTM/GRU/Transformer و مدل‌های volatility نیستند.

**ارتباط با ادبیات:** surveyها مدل‌های متنوع‌تری را بررسی می‌کنند و RNNها در برخی meta-analysisها بهتر گزارش شده‌اند، اما ناهمگونی داده و split مانع انتقال مستقیم نتایج است. [Ryll & Seidens](https://arxiv.org/abs/1906.07786)

**اثر:** کمبود candidateها ممکن است سقف عملکرد را محدود کند، ولی AUC نزدیک 0.5 در چند خانواده نشان می‌دهد تعویض الگوریتم به‌تنهایی مشکل بنیادی را حل نمی‌کند.

**شدت:** متوسط.

## 5. بخش اعتبارسنجی و آماره‌ها

### مشکل 16 — purging وجود دارد، اما embargo و CPCV کامل نیست

**در پروژه:** شرط `label_end_index < validation_start_row` overlap برچسب را کنترل می‌کند، اما embargo مستقل بعد از validation و CPCV/PBO/DSR وجود ندارد.

**ارتباط با ادبیات:** López de Prado و ابزارهای purged CV بر purge، embargo، CPCV، probability of backtest overfitting و deflated Sharpe تأکید می‌کنند. [purgedcv](https://github.com/eslazarev/purged-cross-validation)

**اثر:** label leakage اصلی کنترل شده، اما serial dependence، multiple testing و backtest selection risk به‌طور کامل اندازه‌گیری نشده‌اند.

**شدت:** زیاد.

### مشکل 17 — تعداد fold مؤثر کم و regimeها ناهمگون‌اند

**شاهد:** fold اول قابل استفاده نمانده و خروجی‌ها عمدتاً `wf_02` تا `wf_05` هستند؛ عملکرد fold آخر افت می‌کند.

**اثر:** median چهار fold نمی‌تواند تمام uncertainty بین regimeها را توصیف کند.

**شدت:** زیاد.

### مشکل 18 — locked test برای افق‌های بلند از نظر اقتصادی کوچک است

**شاهد:** در backtest غیرهم‌پوشان، افق 20 فقط 40 معامله دارد.

**اثر:** Sharpe، hit rate و drawdown افق 20 و حتی مقایسه مدل‌ها variance زیادی دارند.

**شدت:** زیاد.

### مشکل 19 — معیار 60% با ادبیات و واقعیت بازار سخت‌گیرانه و احتمالاً نامتناسب است

**در پروژه:** شرط هم‌زمان balanced accuracy ≥ 0.60، macro-F1 ≥ 0.55 و recall هر دو کلاس ≥ 0.50 است.

**اثر:** این معیار برای یک تک‌دارایی nonstationary بدون macro data ممکن است بالاتر از edge پایدار قابل‌انتظار باشد. confidence interval افق 1 نیز حدود `[0.520, 0.581]` است و حد بالا زیر 0.60 می‌ماند.

**شدت:** بحرانی در طراحی thesis acceptance.

## 6. بخش backtest و ادعای کاربردی

### مشکل 20 — backtest با classification target یکسان نیست

**در پروژه:** هر prediction binary به signal `+1/-1` تبدیل می‌شود و هزینه ثابت 5 bps کم می‌شود؛ position sizing، uncertainty، no-trade zone و risk targeting وجود ندارد.

**اثر:** مدل با probability نزدیک 0.5 هم معامله می‌کند، درحالی‌که چنین predictionهایی باید احتمالاً abstain/no-trade باشند.

**شدت:** زیاد.

### مشکل 21 — metric اقتصادی با benchmarkهای کافی مقایسه نشده است

**در پروژه:** backtest summary وجود دارد، اما baselineهای buy-and-hold، always-up، random signal، momentum و volatility-filtered strategy در خروجی اصلی به‌صورت کامل مقایسه نمی‌شوند.

**اثر:** معلوم نیست زیان/سود مدل بهتر یا بدتر از یک استراتژی ساده است.

**شدت:** زیاد.

## 7. بخش reproducibility و ادعای پژوهشی

### مشکل 22 — داده بازار واقعی کاملاً قابل‌تکرار عمومی نیست

**در پروژه:** checksum فایل ثبت شده، اما فایل از MT5 broker export است و metadata broker/server/timezone خالی است.

**اثر:** پژوهشگر دیگر ممکن است نتواند دقیقاً همان candles و locked test را بازسازی کند.

**شدت:** بحرانی برای مقاله.

### مشکل 23 — baselineهای ادبیات و ablation در artifacts رسمی نیستند

**اثر:** contribution Hurst و gate قابل‌اثبات نیست؛ بدون مقایسه با price-only، technical-only، no-Hurst و no-gate نمی‌توان گفت کدام جزء ارزش افزوده ایجاد کرده است.

**شدت:** بحرانی.

### مشکل 24 — نسخه فعلی نتیجه بازار را تأیید نمی‌کند

**شاهد:** هر چهار horizon در locked test `FAIL` هستند؛ افق‌های 10 و 20 در backtest بازده خالص منفی دارند.

**اثر:** پروژه فعلاً یک software/research framework معتبر است، نه یک مدل پیش‌بینی موفق یا strategy profitable.

**شدت:** بحرانی برای claims.

## 8. اولویت‌بندی نهایی

### باید قبل از هر تغییر مدل حل شوند

1. تکمیل provenance: broker، server، timezone، session و decision timestamp.
2. اجرای baseline و ablation رسمی، مخصوصاً no-Hurst و no-gate.
3. بازبینی target و جداسازی statistical direction از actionable direction.
4. اضافه‌کردن و audit متغیرهای کلان طلا یا صریحاً محدودکردن ادعا به univariate OHLCV.
5. بررسی Hurst با estimatorهای robust و چند window.

### باید قبل از ادعای novelty/robustness حل شوند

6. ارزیابی calibration probability و gate در چند fold.
7. گزارش embargo، CPCV/PBO یا توضیح رسمی علت نبود آن‌ها.
8. گزارش عملکرد جداگانه در bull/bear/high-vol/low-vol.
9. مقایسه اقتصادی با baselineهای ساده و no-trade zone.

### بهبودهای مرحله دوم

10. افزودن SVM/XGBoost و deep models به‌عنوان benchmark، نه به‌عنوان راه‌حل قطعی.
11. ساخت expertهای واقعاً regime-specialized.
12. مدل‌سازی horizon-specific به‌جای استفاده تقریباً یکسان از feature set.

## نتیجه نهایی

بزرگ‌ترین مشکل پروژه این نیست که مدل ساده است؛ مشکل این است که پروژه از داده محدود و عمدتاً درون‌قیمتی، target بسیار نویزی و معیار پذیرش سخت‌گیرانه، انتظار edge پایدار 60% دارد. ادبیات gold و financial ML نشان می‌دهد Hurst، regime، deep learning یا gating هیچ‌کدام به‌تنهایی چنین تضمینی نمی‌دهند. قبل از پیچیده‌ترکردن مدل باید ثابت شود هر جزء پروژه، خصوصاً Hurst و gate، در locked out-of-sample به‌طور مستقل ارزش افزوده ایجاد می‌کند.
