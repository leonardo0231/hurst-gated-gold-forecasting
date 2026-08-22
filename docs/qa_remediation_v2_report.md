# گزارش canonical اصلاح QA و اجرای پژوهشی v2

تاریخ: 2026-08-22  
خانواده: `executable_direction_hurst_ablation_v2`  
run نهایی: `executable_direction_hurst_ablation_v2-20260822T165905Z`  
نتیجه علمی: **هر ۱۲ آزمایش توسعه رد شدند؛ هیچ candidate فریز نشد.**

## ۱. پروتکل پیاده‌سازی‌شده

خانوادهٔ v1 و تمام آرتیفکت‌های آن بدون تغییر حفظ شد. خانوادهٔ مستقل v2 پیش از اجرا با
بودجهٔ ثابت ۱۲ آزمون (سه بازوی no-Hurst، DFA Hurst و robust Hurst در چهار افق ۱، ۵، ۱۰ و
۲۰) فریز شد. کارت فرضیه، تنظیمات اجرایی، عضویت دقیق ویژگی‌ها، کد runtime، dependency،
داده و manifestها SHA-256 شدند. اجرا از staging به final به‌صورت atomic منتقل شد، registry
با lock بین‌پردازه‌ای و hash chain تکمیل شد و شکست‌ها دارای failed-run receipt هستند.

schedule اقتصادی v2 فقط اولین سیگنال واجد شرایط هنگام flat بودن را می‌پذیرد؛ سیگنال ردشدهٔ
هم‌پوشان `busy_until` را تغییر نمی‌دهد و ورود در همان open خروج قبلی مجاز است. مسیر بازده
کامل هر observation، پنج benchmark روی schedule یکسان، هزینهٔ پایهٔ ۵ bps و stress ده bps
ذخیره شده است.

## ۲. پارتیشن‌ها و وضعیت exposure

| طبقه شواهد | مرز واقعی | وضعیت و استفاده |
|---|---|---|
| توسعه | 2011-01-03 تا 2023-06-30، ۳۲۲۱ ردیف | `development_reused_previously_exposed`؛ nested development فقط |
| confirmation تاریخی | وجود ندارد | 2022 تا June 2023 قبلاً در foldها/نتایج دیده شده و confirmation نیست |
| audit تاریخی | 2023-07-03 تا 2026-07-31، ۷۹۶ ردیف | `historical_audit_previously_revealed`؛ در v2 بارگذاری نشد |
| آیندهٔ واقعی | هنوز وجود ندارد | تنها شواهد future OOS پس از freeze یک candidate آینده خواهد بود |

هش development برابر
`e5d90295a128fc37924f6d144a2f03701caf78490152335bcfe82311d974adfa` و هش audit برابر
`b6998ff686cbdd2475003340f90c06498b77fe22ca354e9f23b36a87d228fb31` است. دو فایل از
نظر تاریخ جدا هستند و ترکیبشان منبع ۴۰۱۷ ردیفی legacy را دقیقاً بازسازی می‌کند.

## ۳. inventory آزمایش‌های توسعه

| بازو | H | BA تجمیعی | macro-F1 | net log return 5bps | net log return 10bps | تصمیم |
|---|---:|---:|---:|---:|---:|---|
| no-Hurst | 1 | 0.52035 | 0.51618 | -0.59011 | -1.12071 | رد |
| no-Hurst | 5 | 0.49252 | 0.48921 | -0.36571 | -0.59904 | رد |
| no-Hurst | 10 | 0.49906 | 0.49393 | -0.30315 | -0.37260 | رد |
| no-Hurst | 20 | 0.47561 | 0.43109 | -0.16283 | -0.19831 | رد |
| DFA Hurst | 1 | 0.50537 | 0.47193 | -0.72217 | -1.23328 | رد |
| DFA Hurst | 5 | 0.48261 | 0.48078 | -0.22915 | -0.41351 | رد |
| DFA Hurst | 10 | 0.48270 | 0.47941 | -0.05765 | -0.08463 | رد |
| DFA Hurst | 20 | 0.48016 | 0.42692 | -0.06271 | -0.09918 | رد |
| robust Hurst | 1 | 0.49490 | 0.49474 | -0.93280 | -1.79066 | رد |
| robust Hurst | 5 | 0.50906 | 0.50906 | 0.40854 | 0.17421 | رد |
| robust Hurst | 10 | 0.48640 | 0.48534 | 0.15860 | 0.05218 | رد |
| robust Hurst | 20 | 0.51239 | 0.46546 | -0.03111 | -0.07857 | رد |

بهترین BA متعلق به no-Hurst/H1 است؛ CI بوت‌استرپ اسمی ۹۵٪ آن `[0.50189, 0.53837]`،
macro-F1 آن 0.51618 و هر دو سناریوی هزینه منفی است. این trial در معیار median BA، ۴ از ۵
fold، macro-F1، recall صعود، calibration، CI بازده جفتی، اقتصاد، PBO، DSR و QA شکست خورد.

## ۴. نتیجهٔ Hurst/no-Hurst

هیچ شواهد معتبر و پایدار برای ارزش افزودهٔ Hurst وجود ندارد. بازوی no-Hurst بهترین BA کل
خانواده را دارد. robust-Hurst/H5 و H10 بازده نقطه‌ای مثبت دارند، ولی معیارهای طبقه‌بندی،
عدم‌قطعیت، calibration، PBO/DSR و QA را پاس نمی‌کنند. افزون بر آن، v2 فاصلهٔ paired و CI
مستقیم Hurst در برابر no-Hurst را gate نکرده است؛ بنابراین حتی عبارت «بهبود افزایشی Hurst»
مجاز نیست.

## ۵. اقتصاد، confirmation، audit و MT5

هیچ candidate از development عبور نکرد؛ در نتیجه confirmation، audit تاریخی جدید و MT5
candidate backtest مجاز نبود و اجرا نشد. مثبت بودن نقطه‌ای robust-Hurst/H5/H10 اثبات سودآوری
نیست. قرارداد reconciliation برای candidate آینده، نگاشت ردیف‌به‌ردیف
`signal → order → entry fill → exit fill → spread/commission/swap/slippage` را الزام می‌کند.
این قرارداد replay سیگنال است و `native_mql5_inference=false`؛ parity واقعی terminal تا وجود
candidate فریز‌شده deferred است.

## ۶. leakage و بازتولیدپذیری

- loader v2 فقط فایل فیزیکی development را با role، مرز، تعداد ردیف و hash معتبر می‌خواند.
- outer/inner foldهای اصلی بر `executable_label_end_index` purge می‌شوند و transformها فقط
  روی train fit می‌شوند؛ بااین‌حال مرز ثانویهٔ calibration/evaluation در inner OOF این
  endpoint را purge نکرده است و ادعای leakage-free برای v2 رد می‌شود.
- receipt نهایی ۱۱۰ خروجی و ۲۱ منبع runtime را پوشش می‌دهد و دوباره validate شد.
- registry شامل دقیقاً ۱۲ رکورد و head hash
  `183fb0604debc7eca7791b96d1de331e9ca24acb50d1903a963dda0f7d4074b6` است.
- metadata receipt صریحاً `historical_audit_accessed=false` را ثبت می‌کند.
- run ناقص `151330Z` حذف نشد؛ با هفت فایل و manifest هش‌شده به quarantine منتقل شد.

## ۷. محدودیت‌ها و انحراف‌های کشف‌شده پس از freeze

v2 به‌طور immutable ثبت می‌کند که runner هیچ candidate را promote نکرد، اما metrics آن به‌علت
نقص calibration-boundary برای استنباط promotion/rejection مستقل یا ادعای leakage-free کافی
نیست:

1. در هر ۶۰ انتخاب outer، انتهای برچسب calibration از شروع evaluation عبور می‌کند؛ sigmoid
   در ۲۶ انتخاب برگزیده شده و این leakage بر OOF نهایی اثر بالقوه دارد.
2. مرز outer foldها بعد از eligibility هر horizon جدا ساخته شده و common-calendar کامل نیست.
3. CI طبقه‌بندی per-trial اسمی است و familywise correction برای ۱۲ trial ندارد.
4. PBO در boundary ترکیب‌ها event چندروزهٔ crossing را purge نمی‌کند و DSR dependence-robust
   نیست؛ این دو فقط diagnostic هستند.
5. Hurst با paired block CI مستقیم در برابر no-Hurst gate نشده است.
6. طول block برای هر horizon تطبیقی نیست، ECE گزارش‌شده بدون وزن است و همهٔ intervalهای
   bootstrap persist نشده‌اند.
7. provenance ساعت/سشن منبع legacy بازسازی‌شده است و `provenance_complete=false` عمداً مانع
   promotion می‌شود.

طبق candidate freeze policy هیچ‌کدام پس از دیدن v2 در همان خانواده اصلاح یا rerun نمی‌شوند.
هر اصلاح آماری بعدی به کارت، شناسه و بودجهٔ مستقل نیاز دارد.

## ۸. موارد deferred

fresh future confirmation، familywise classification inference، common-calendar folds،
event-boundary-safe PBO/DSR، paired Hurst inference، point-in-time macro data، provenance زمان
بومی، cross-broker/real-tick MT5 و native MQL5 inference در
`artifacts/research/deferred_items.jsonl` ثبت شده‌اند.

همچنین ممیزی پروتکل نشان داد `freeze_candidate()` با وجود حفاظت creation-exclusive و hash، eligibility را هنوز
مستقیماً از registry/finalization/QA resolve نمی‌کند و به boolean فراخواننده اعتماد دارد. چون هیچ candidateای وجود ندارد و
v2 immutable است، این کنترل به‌عنوان P1 پیش از هر candidate آینده ثبت شد و نباید پسینی به v2 افزوده شود.

## ۸-۱. جمع‌بندی پنج عامل مستقل

| عامل | اولویت/یافته اصلی | فایل‌های درگیر | ریسک و اقدام | وضعیت |
|---|---|---|---|---|
| research protocol | P0: بازه 2022 تا ژوئن 2023 قبلاً در fold پنجم و انتخاب استفاده شده است. | baseline/data manifests، partitions، protocol | confirmation نامیدن آن ممنوع؛ فقط development reused | پیاده‌سازی‌شده |
| data/feature | P1: پارتیشن فیزیکی معتبر است، ولی provenance زمان/session منبع legacy بازسازی‌شده و volume از نوع tick-volume کارگزار است. | source/availability manifests، partition tests | ادعای point-in-time کامل ممنوع؛ منابع macro فقط با availability واقعی | بخشی پیاده؛ داده بیرونی deferred |
| validation/statistics | P0/P1: common-calendar، familywise CI، event-boundary PBO/DSR، paired Hurst و block تطبیقی ناقص‌اند. | splitter/statistical artifacts | برای ادعای promotion/Hurst کافی نیست؛ خانواده جدید لازم است. | deferred |
| model/experiment | P0: هر ۶۰ مرز calibration/evaluation endpoint purge نشده؛ sigmoid در ۲۶ انتخاب برگزیده شد. | v2 runner و inner artifacts | v2 leakage-free/promotion-valid نیست؛ rerun همان شناسه ممنوع | deferred به خانواده جدید |
| independent QA | P0: ۱۲/۱۲ رد و هیچ candidate/confirmation/audit/MT5 candidate run وجود ندارد. | همه evidenceهای بالا | رأی انتشار مدل `reject`؛ فقط انتشار نتیجه منفی با محدودیت صریح | اجرا و ثبت شد |

گزارش مستقل نهایی در `docs/independent_qa_remediation_v2.md` و نسخه ماشین‌خوان آن در
`artifacts/research/qa/independent_qa_remediation_v2.json` قرار دارد.

## ۹. فرمان‌های اصلی و نتیجه

```powershell
uv run ruff check <changed research files>             # PASS
uv run mypy src                                        # PASS؛ 24 فایل
uv run pytest -q                                       # PASS؛ 82 تست
uv run python scripts/freeze_research_partitions.py --project-root . --boundary 2023-07-03
uv run python scripts/freeze_research_family_v2.py --project-root .
uv run python scripts/run_research_batch_v2.py --project-root .  # PASS؛ 579.2s
```

`ruff check .` روی کل workspace پاس نشد، زیرا `.docx_work` شامل اسکریپت‌های موقتِ کاربر و
چند فایل legacy دارای بدهی format است؛ lint متمرکز همهٔ فایل‌های پژوهشی تغییرکرده پاس شد.

دو DOCX canonical با Microsoft Word به PDF صادر و همه ۵۴ صفحه پایان‌نامه و ۱۴ صفحه گزارش به PNG render و
بازبینی بصری شدند. هر دو فایل نیز به‌عنوان ZIP/DOCX باز شدند و ساختارشان سالم بود؛ نسخه‌های اصلی کاربر overwrite نشدند.

## ۱۰. Skills Used

- `systematic-debugging`: بازتولید ریشهٔ خطای schedule هم‌پوشان و جداسازی علت از علامت.
- `test-driven-development`: ثبت تست قرمز برای schedule، receipt/lock، partition، registry و
  زنجیرهٔ MT5 و سپس سبزکردن آن‌ها.
- `verification-before-completion`: جلوگیری از ادعای موفقیت پیش از receipt، hash validation،
  lint، type-check و test.
- `documents`: برای ممیزی و به‌روزرسانی نسخه‌های Word پایان‌نامه/گزارش با حفظ نسخه‌های اصلی؛
  خروجی‌های canonical صفحه‌به‌صفحه render و از نظر بریدگی/خرابی بررسی شدند.

از skillهای GitHub، Hugging Face، Jupyter و Vercel استفاده نشد؛ این پروژه برای این اصلاح به
repository/CI خارجی، dataset عمومی HF، notebook یا deployment وب نیاز نداشت.
