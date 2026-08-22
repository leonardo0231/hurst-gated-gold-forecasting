# مرور ادبیات یادگیری ماشین در بازار مالی و تطبیق با HGE Gold Forecasting

**تاریخ بررسی:** 2026-08-16  
**مخزن بررسی‌شده:** `hurst-gated-gold-forecasting`، شاخه محلی `thesis-v2-rebuild`  
**دامنه:** پیش‌بینی سری زمانی مالی، پیش‌بینی جهت طلا/XAUUSD، Hurst و حافظه بلندمدت، regime switching، ensemble/gating، اعتبارسنجی زمانی و کنترل leakage، به‌همراه منابع کد و مدل‌های قابل‌تکرار.

## 1. نتیجه اجرایی

نزدیک‌ترین اثر علمی به ایده اصلی پروژه، مقاله **Yang, Wang, Zeng & Li (2024)** با عنوان *Improved prediction of global gold prices: An innovative Hurst-reconfiguration-based machine learning approach* است. آن مقاله نیز طلا، تحلیل Hurst و ML را در یک چارچوب ترکیبی جمع می‌کند و گزارش می‌دهد مدل Hurst-based آن از مدل‌های متعارف از نظر خطای پیش‌بینی و دقت جهت بهتر بوده است. بااین‌حال، آن کار «Hurst-reconfiguration/decomposition + swarm optimization + ensemble» است؛ پروژه حاضر از **rolling Hurst به‌عنوان feature/regime signal** استفاده می‌کند و با **learned regime gate** بین مدل‌های پایه انتخاب می‌کند. شباهت مفهومی زیاد است، اما از روی شواهد موجود نمی‌توان گفت کد پروژه از آن مقاله کپی شده است.

پروژه فعلی بیش از آن‌که یک ادعای «مدل برتر بازار» باشد، یک pipeline پژوهشیِ leakage-safe است: OHLCV، هدف binary جهت در افق‌های 1/5/10/20 روز، featureهای تکنیکال/نوسان/entropy/volume/Hurst، locked test، purged walk-forward، چند مدل پایه، gate و bootstrap CI. خروجی واقعی فعلی این ادعا را تأیید نمی‌کند: هر چهار horizon در `locked_test_metrics.csv` وضعیت `FAIL` دارند و balanced accuracy به‌ترتیب حدود 0.551، 0.523، 0.483 و 0.476 است.

## 2. تاریخچه فشرده و خط زمانی

| سال | اثر/نویسنده | ایده اصلی و اهمیت | ارتباط با پروژه |
|---|---|---|---|
| 1988 | Halbert White، *Economic prediction using neural networks: the case of IBM daily stock returns* | از نخستین کاربردهای مستقیم NN برای بازده روزانه IBM؛ تأکید بر غیرخطی‌بودن و نیاز به استنباط آماری در داده اقتصادی. | ریشه تاریخی مسئله «پیش‌بینی بازده با ML»، اما نه طلا و نه Hurst. |
| 1995–1996 | Kuan & Liu؛ Tenti؛ Verkooijen | مقایسه شبکه‌های feed-forward/recurrent با مدل‌های خطی برای ارز؛ بررسی out-of-sample و هزینه معامله. | پشتیبان استفاده از ارزیابی زمانی و مقایسه مدل‌ها؛ پروژه فعلی RNN ندارد. |
| 1997 | Hochreiter & Schmidhuber، LSTM | حافظه بلندمدت در شبکه‌های recurrent. | از نظر موضوعی به long memory نزدیک است، ولی پروژه حاضر مدل‌های کلاسیک sklearn دارد. |
| 2003 | Kyoung-jae Kim، *Financial time series forecasting using support vector machines* | SVM در برابر BP و CBR برای شاخص؛ ادعای SVM به‌عنوان گزینه مناسب برای سری مالی. | نشان می‌دهد مقایسه چند مدل و regularization مهم است؛ SVM در نسخه فعلی پروژه نیست. |
| 2004 | Qian & Rasheed، *Hurst Exponent and Financial Market Predictability* | سری‌های با Hurst بالاتر با BP neural network دقیق‌تر پیش‌بینی شدند؛ H به‌عنوان proxy قابلیت پیش‌بینی پیشنهاد شد. | یکی از مستقیم‌ترین ریشه‌های ایده Hurst-regime؛ پروژه آن را با rolling R/S و gate عملیاتی می‌کند. |
| 2007 | Eom, Choi, Oh & Jung | رابطه Hurst و hit-rate جهت در 60 شاخص؛ H بالاتر با predictability بیشتر همراه گزارش شد. | پشتیبان نظری استفاده از Hurst برای conditioning، ولی بازار و روش پروژه متفاوت است. |
| 2009–2012 | مطالعات ensemble، SVM، ANN و hybrid در stock/FX | ترکیب مدل‌ها و feature selection برای غیرایستایی و نویز. | زمینه انتخاب Logistic/RF/ExtraTrees/HGB و ساخت gate. |
| 2018 | Marcos López de Prado، *Advances in Financial Machine Learning* | purging، embargo، CPCV، backtest overfitting و اهمیت label intervals. | مستقیم‌ترین منبع روش‌شناختی برای کنترل overlap/leakage؛ پروژه نسخه ساده‌تر purged walk-forward دارد. |
| 2019 | Ryll & Seidens، survey/meta-analysis بیش از 150 مطالعه | ناهمگونی شدید داده/معیارها را نشان می‌دهد؛ RNNها در میانگین از feed-forward و SVM بهتر گزارش شدند، اما مقایسه بین مقاله‌ها را محدود می‌داند. | هشدار مهم برای تفسیر هر «دقت 60%» و علت تمرکز پروژه بر پروتکل ارزیابی. |
| 2020–2022 | Jiang؛ Sezer et al. و surveyهای DL مالی | رشد LSTM/CNN/attention/transformer و نقد reproducibility و backtesting. | پروژه عمداً از deep modelها صرف‌نظر کرده تا auditability و reproducibility حفظ شود. |
| 2023–2024 | Yang, Wang, Zeng & Li؛ *Improved prediction of global gold prices* | Hurst-oriented reconfiguration روی سه بازار طلا؛ تفاوت multifractal بازارها؛ رابطه منفی خطای پیش‌بینی و Hurst؛ برتری نسبت به مدل‌های متعارف ادعا شده. | نزدیک‌ترین مقاله مستقیم؛ similarity بالا در gold + Hurst + ML، تفاوت جدی در decomposition/embedding/swarm و نبود learned gate مشابه. |
| 2025 | مطالعات جدید XAU/USD و ML | مقایسه indicatorها و RF/GB/XGBoost؛ نتایج معمولاً حوالی 55–60% و وابسته به split/cost. | benchmark زمینه‌ای؛ دقت‌های کوچک بالای 50% بدون locked test و هزینه، evidence قوی محسوب نمی‌شوند. |
| 2026 | *Hybrid Recurrent Expert Gating*؛ Procedia Computer Science | MoE با RNN/LSTM/GRU و softmax gating روی OHLCV سهام Google؛ بهبود نسبت به GRU در MAE/MSE/RMSE/R² گزارش شده. | از نظر «gating بین expertها» نزدیک است، اما asset، loss، مدل‌ها و معماری با HGE متفاوت‌اند؛ شاهدی از کپی‌بودن نیست. |

## 3. آثار کلیدی با اطلاعات نویسندگان، دستاورد و محدودیت

### 3.1 White (1988)

**عنوان:** *Economic prediction using neural networks: the case of IBM daily stock returns*  
**نویسنده:** Halbert White.  
**دستاورد:** کاربرد شبکه عصبی برای کشف regularityهای غیرخطی در بازده روزانه IBM و برجسته‌کردن مسائل خاص داده اقتصادی.  
**محدودیت برای استفاده در پایان‌نامه:** قدیمی، تک‌دارایی، و فاقد استانداردهای امروزی برای purged validation، هزینه، چند افق و locked test.

### 3.2 Qian & Rasheed (2004)

**عنوان:** *Hurst Exponent and Financial Market Predictability*  
**نویسندگان:** B. Qian و K. Rasheed.  
**دستاورد:** H=0.5 به‌عنوان حالت تصادفی، H>0.5 به‌عنوان persistence/trend؛ شبکه BP روی سری‌های با H بزرگ‌تر عملکرد بهتری نشان داد.  
**ارتباط:** پروژه حاضر همین شهود را از حالت «طبقه‌بندی سری‌ها» به featureهای rolling (`hurst_rs_64`, `hurst_rs_128`) و `hurst_regime` تبدیل می‌کند.  
**احتیاط:** Hurst estimator نسبت به window، nonstationarity و روش برآورد حساس است؛ این مقاله مجوز ادعای causal predictive power برای طلا نیست.

### 3.3 Eom et al. (2007)

**عنوان:** *Hurst exponent and prediction based on weak-form efficient market hypothesis of stock markets*  
**نویسندگان:** Cheoljun Eom، Sunghoon Choi، Gabjin Oh، Woo-Sung Jung.  
**دستاورد:** مطالعه 60 شاخص و گزارش رابطه مثبت Hurst و hit-rate جهت؛ استفاده از nearest-neighbor برای prediction.  
**ارتباط:** توجیه literature-level برای conditioning بر Hurst.  
**تفاوت:** stock indexes و nearest-neighbor، نه XAUUSD OHLCV و نه expert gate.

### 3.4 Kim (2003)

**عنوان:** *Financial time series forecasting using support vector machines*  
**نویسنده:** Kyoung-jae Kim.  
**دستاورد:** مقایسه SVM با BP و case-based reasoning برای شاخص؛ برجسته‌کردن structural risk minimization و کنترل overfit.  
**ارتباط:** پشتوانه مقایسه baselineهای ساده و nonlinear ensembleها.  
**محدودیت:** پروتکل‌های قدیمی‌تر و عدم انطباق مستقیم با multi-horizon gold.

### 3.5 Yang et al. (2024)

**عنوان:** *Improved prediction of global gold prices: An innovative Hurst-reconfiguration-based machine learning approach*  
**نویسندگان:** Mo Yang، Ruotong Wang، Zixun Zeng، Peizhi Li.  
**ناشر/شناسه:** *Resources Policy*, 88, 104430؛ DOI: `10.1016/j.resourpol.2023.104430`.  
**دستاوردهای گزارش‌شده:** تحلیل سه بازار عمده طلا؛ تفاوت ساختارهای multifractal؛ رابطه منفی forecasting error و Hurst؛ رابطه منفی embedding dimension و Hurst؛ برتری مدل hybrid از نظر خطا و direction accuracy نسبت به مدل‌های متعارف.  
**شباهت به پروژه:** بسیار زیاد در سه کلمه کلیدی gold، Hurst، machine learning.  
**تفاوت تعیین‌کننده:** کار آن‌ها Hurst-oriented reconfiguration/decomposition و ensemble با swarm optimization است؛ پروژه حاضر از featureهای causal، regimeهای rolling، مدل‌های sklearn و learned logistic gate استفاده می‌کند.  
**نتیجه تطبیق:** «الهام/هم‌راستایی علمی محتمل»؛ «استفاده مستقیم از کد» اثبات‌نشده؛ «همان پروژه» نیست.

### 3.6 Ryll & Seidens (2019)

**عنوان:** *Evaluating the Performance of Machine Learning Algorithms in Financial Market Forecasting: A Comprehensive Survey*  
**نویسندگان:** Lukas Ryll و Sebastian Seidens.  
**دستاورد:** جمع‌آوری بیش از 150 مطالعه؛ rank analysis بین الگوریتم‌ها؛ گزارش میانگین بهتر RNN نسبت به feed-forward و SVM.  
**نکته بسیار مهم:** خود مقاله می‌گوید داده، asset، معیار و روش آزمایش بین مطالعات استاندارد نیست و مقایسه مستقیم سخت است.  
**ارتباط:** از نظر روش گزارش‌دهی و هشدار reproducibility برای پروژه مهم‌تر از انتخاب یک مدل خاص است.

### 3.7 López de Prado (2018) و purged CV

**ایده:** در labelهایی که به چند روز آینده وابسته‌اند، نمونه‌های train و validation ممکن است بازه برچسب مشترک داشته باشند؛ purge و embargo برای حذف leakage ضروری‌اند.  
**در پروژه:** `label_end_index < validation_start_row` برای train اعمال شده، تست نهایی زمانی جدا است و انتخاب model/threshold/gate از locked test استفاده نمی‌کند.  
**تفاوت:** پروژه HGE فعلی walk-forward purged دارد، اما CPCV، PBO، deflated Sharpe و embargo مستقلِ قابل‌تنظیم را به‌طور کامل پیاده نکرده است.

### 3.8 کارهای جدید regime/gating

کارهای 2021 به بعد regime detection را با clustering/classification و کارهای 2026 با Mixture-of-Experts و softmax routing دنبال کرده‌اند. اثر 2026 روی Google OHLCV، LSTM/GRU/RNN را با gating ترکیب می‌کند و بهبودهای کوچک تا متوسط در خطا گزارش می‌دهد. این بدنه ادبیات از **اصل انتخاب expert بر اساس regime** پشتیبانی می‌کند، اما مستقیماً ادعا نمی‌کند Hurst gate پروژه حاضر همان معماری یا همان نتیجه را دارد.

## 4. ماتریس تطبیق با کد فعلی پروژه

| جزء پروژه | شاهد در مخزن | نزدیک‌ترین بدنه ادبیات | وضعیت تطبیق |
|---|---|---|---|
| هدف `sign(future log return)` در افق 1/5/10/20 | `src/hge_gold/targets.py`, `docs/thesis_plan.md` | مطالعات directional classification مالی | الگوی استاندارد؛ نه ایده اختصاصی یک مقاله |
| OHLCV و featureهای lag/rolling | `src/hge_gold/features.py` | technical-indicator ML literature | استفاده از خانواده‌های شناخته‌شده؛ فرمول‌بندی ترکیبی پروژه‌ای |
| RSI, ATR, MACD, skew/kurtosis, volume | `features.py` | technical analysis + ML | مشابه مفهومی گسترده؛ citation تک‌منبعی قابل انتساب نیست |
| rolling R/S Hurst در windowهای 64 و 128 | `_hurst_rs`, `hurst_rs_*` | Qian & Rasheed؛ Eom et al.؛ Yang et al. | شباهت مستقیم مفهومی؛ estimator پروژه ساده R/S است |
| `hurst_regime` با quantileهای rolling و `shift(1)` | `features.py` | Hurst-based regime/predictability | نوآوری اجرایی پروژه؛ از leakage جلوگیری می‌کند |
| `trend_regime`, volatility regime | `features.py` | regime-switching و adaptive forecasting | ترکیب شناخته‌شده، پیاده‌سازی خاص پروژه |
| Logistic/RF/ExtraTrees/HGB candidates | `modeling.py` | surveyهای مقایسه ML مالی | انتخاب baseline/ensemble متعارف |
| OOF probability + learned logistic gate | `modeling.py` | MoE/ensemble/regime gating | مشابه معماری gating؛ نسخه پروژه shallow و interpretable است |
| fallback به best base | `modeling.py` | model selection/robust ensemble | تصمیم مهندسی محافظه‌کارانه؛ به‌صورت دقیق در منابع بررسی‌شده پیدا نشد |
| locked chronological test | `splits.py`, `pipeline.py` | out-of-sample financial forecasting | استاندارد ضروری، نه ادعای نوآوری مستقل |
| purged walk-forward | `splits.py` | López de Prado و پیاده‌سازی‌های purged CV | تطبیق قوی؛ CPCV/PBO هنوز اضافه نشده |
| bootstrap moving block CI | `evaluation.py` | dependence-aware inference | روش مناسب برای سری وابسته؛ باید با baseline و multiple testing تکمیل شود |
| cost/slippage research backtest | `evaluation.py`, config | financial backtesting | مثبت؛ هنوز evidence سرمایه‌گذاری واقعی نیست |
| audit، provenance، hash و testهای causality | `pipeline.py`, tests | reproducibility/MLOps for research | وجه متمایز پروژه از بسیاری از notebookهای بازار |

## 5. منابع GitHub، Hugging Face و نرم‌افزار

1. **مخزن خود پروژه:** [leonardo0231/hurst-gated-gold-forecasting](https://github.com/leonardo0231/hurst-gated-gold-forecasting). صفحه عمومی GitHub در زمان بررسی، شاخه `main` را به‌عنوان چارچوب گسترده‌تر HGE-Hybrid با فازهای 0–11 نشان می‌دهد؛ شاخه محلی بررسی‌شده `thesis-v2-rebuild` نسخه باریک‌تر V2 است. بنابراین «مخزن عمومی» و «کد checkout‌شده» دقیقاً یک snapshot نیستند.
2. **purged-cross-validation:** [eslazarev/purged-cross-validation](https://github.com/eslazarev/purged-cross-validation). قابلیت‌های purge، embargo، walk-forward، CPCV، PBO، PSR و DSR را ارائه می‌کند؛ برای مقایسه یا ارتقای protocol پروژه مفید است، اما شاهدی از استفاده‌شدن در کد فعلی نیست.
3. **Gold forecasting example:** [HuguitoH/gold-forecast](https://github.com/HuguitoH/gold-forecast). پروژه XAU/USD با yfinance، XGBoost، walk-forward، featureهای macro و dashboard است؛ از نظر asset و walk-forward مشابه، ولی Hurst gate و audit protocol پروژه حاضر را ندارد.
4. **Hugging Face:** در جست‌وجوی هدفمند فعلی، موردی که هم‌زمان Hurst، XAUUSD و معماری gate پروژه حاضر را بازتولید کند پیدا نشد. نبود نتیجه در HF به معنای نبود هیچ dataset/model مرتبطی نیست؛ برای ادعای کامل باید API/Hub با queryهای چندزبانه و pagination کامل نیز crawl شود.

## 6. مواردی که پروژه احتمالاً از آن‌ها استفاده نکرده است

- LSTM/GRU/Transformer/CNN به‌عنوان مدل اصلی.
- sentiment/news/NLP و macro cross-asset features.
- GARCH؛ در اسناد پروژه صراحتاً deferred است.
- SVM و XGBoost در نسخه محلی فعلی.
- CPCV، PBO، deflated Sharpe و probabilistic Sharpe به‌صورت کامل.
- Hurst-based decomposition/multifractal reconstruction مقاله Yang et al.

## 7. نتیجه‌گیری درباره «همان مقاله بودن»

با شواهد فعلی، پروژه **همان مقاله Yang et al. نیست** و کد آن مقاله نیز در مخزن پیدا نشده است. پروژه یک ترکیب مستقل از ایده‌های جاافتاده است: پیش‌بینی جهت طلا، featureهای تکنیکال، rolling Hurst، regime conditioning، ensemble selection و اعتبارسنجی leakage-safe. بیشترین ریسک علمی، نه سرقت ادبی آشکار، بلکه **کمبود citation صریح برای منشأ هر جزء و فاصله بین ادعای هدف 60% و عملکرد locked test فعلی** است.

برای نسخه نهایی پایان‌نامه پیشنهاد می‌شود در بخش related work صریحاً این چهار خانواده تفکیک شوند: (الف) تاریخچه ML مالی، (ب) Hurst و predictability، (ج) gold forecasting، (د) validation/backtest integrity. همچنین باید نتیجه فعلی به‌صورت «عدم عبور از معیار پذیرش» گزارش شود، نه موفقیت بازار.

## 8. فهرست منابع منتخب و پیوند مستقیم

- [White (1988), DBLP record](https://dblp.org/rec/conf/icnn/White88.html)
- [Tenti (1996), Forecasting foreign exchange rates using recurrent neural networks](https://doi.org/10.1080/088395196118434)
- [Kim (2003), Financial time series forecasting using support vector machines](https://doi.org/10.1016/S0925-2312(03)00372-2)
- [Qian & Rasheed (2004), Hurst Exponent and Financial Market Predictability](https://m.actapress.com/Abstract.aspx?paperId=17650)
- [Eom et al. (2007), Hurst exponent and prediction](https://arxiv.org/abs/0712.1624)
- [Ryll & Seidens (2019), comprehensive survey](https://arxiv.org/abs/1906.07786)
- [Jiang, Applications of deep learning in stock market prediction](https://arxiv.org/abs/2003.01859)
- [Yang et al. (2024), Hurst-reconfiguration gold forecasting](https://doi.org/10.1016/j.resourpol.2023.104430)
- [HGE GitHub repository](https://github.com/leonardo0231/hurst-gated-gold-forecasting)
- [purged-cross-validation GitHub](https://github.com/eslazarev/purged-cross-validation)
- [Hybrid Recurrent Expert Gating (2026)](https://doi.org/10.1016/j.procs.2026.06.366)

## 9. محدودیت این گزارش

این گزارش یک مرور هدفمند و evidence-based است، نه ادعای crawl کامل همه صفحات اینترنت، Google Scholar، Scopus، Web of Science، Crossref، arXiv، GitHub و Hugging Face. موتورهای جست‌وجو پوشش و رتبه‌بندی یکسان ندارند، بسیاری از مقالات paywalled هستند و شمارش citationها به تاریخ دسترسی وابسته است. برای تبدیل این خروجی به systematic review قابل‌دفاع باید queryهای ثبت‌شده، PRISMA، exportهای bibliographic، deduplication، غربال عنوان/چکیده و فایل CSV همه رکوردها اجرا شود.
