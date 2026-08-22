#property copyright "HGE Gold Forecasting research project"
#property version   "1.00"
#property strict

#include <Trade/Trade.mqh>

input string InpSignalFile = "hge_signals.csv";
input double InpLots = 0.01;
input ulong InpMagic = 26082026;
input int InpMaxDeviationPoints = 20;

CTrade trade;
datetime entry_times[];
datetime exit_times[];
int directions[];
string signal_ids[];
int next_signal = 0;
datetime last_bar_time = 0;

bool LoadSignals()
{
   int handle = FileOpen(InpSignalFile, FILE_READ | FILE_CSV | FILE_ANSI | FILE_COMMON, ',');
   if(handle == INVALID_HANDLE)
   {
      PrintFormat("SIGNAL_FILE_OPEN_FAILED file=%s error=%d", InpSignalFile, GetLastError());
      return false;
   }

   // Header: entry_time,exit_time,direction,signal_id
   for(int column = 0; column < 4 && !FileIsEnding(handle); column++)
      FileReadString(handle);

   datetime previous_entry = 0;
   while(!FileIsEnding(handle))
   {
      string entry_text = FileReadString(handle);
      if(entry_text == "" && FileIsEnding(handle))
         break;
      string exit_text = FileReadString(handle);
      int direction = (int)FileReadNumber(handle);
      string signal_id = FileReadString(handle);
      datetime entry_time = StringToTime(entry_text);
      datetime exit_time = StringToTime(exit_text);

      if(entry_time <= 0 || exit_time <= entry_time || (direction != -1 && direction != 1))
      {
         PrintFormat("INVALID_SIGNAL id=%s entry=%s exit=%s direction=%d",
                     signal_id, entry_text, exit_text, direction);
         FileClose(handle);
         return false;
      }
      if(previous_entry >= entry_time)
      {
         PrintFormat("NON_MONOTONE_OR_DUPLICATE_SIGNAL id=%s", signal_id);
         FileClose(handle);
         return false;
      }

      int size = ArraySize(entry_times);
      ArrayResize(entry_times, size + 1);
      ArrayResize(exit_times, size + 1);
      ArrayResize(directions, size + 1);
      ArrayResize(signal_ids, size + 1);
      entry_times[size] = entry_time;
      exit_times[size] = exit_time;
      directions[size] = direction;
      signal_ids[size] = signal_id;
      previous_entry = entry_time;
   }
   FileClose(handle);
   PrintFormat("SIGNALS_LOADED count=%d file=%s", ArraySize(entry_times), InpSignalFile);
   return ArraySize(entry_times) > 0;
}

bool HasOurPosition()
{
   if(!PositionSelect(_Symbol))
      return false;
   return (ulong)PositionGetInteger(POSITION_MAGIC) == InpMagic;
}

void HandleNewBar(const datetime bar_time)
{
   if(HasOurPosition())
   {
      string comment = PositionGetString(POSITION_COMMENT);
      int open_index = next_signal - 1;
      if(open_index >= 0 && open_index < ArraySize(exit_times) && bar_time >= exit_times[open_index])
      {
         if(!trade.PositionClose(_Symbol))
            PrintFormat("CLOSE_FAILED id=%s retcode=%u", comment, trade.ResultRetcode());
         else
            PrintFormat("CLOSE_OK id=%s time=%s price=%.5f", comment,
                        TimeToString(bar_time, TIME_DATE | TIME_MINUTES), trade.ResultPrice());
      }
   }

   while(next_signal < ArraySize(entry_times) && entry_times[next_signal] < bar_time)
   {
      PrintFormat("STALE_SIGNAL_SKIPPED id=%s", signal_ids[next_signal]);
      next_signal++;
   }

   if(HasOurPosition() || next_signal >= ArraySize(entry_times))
      return;
   if(entry_times[next_signal] != bar_time)
      return;

   string id = signal_ids[next_signal];
   bool opened = directions[next_signal] > 0
      ? trade.Buy(InpLots, _Symbol, 0.0, 0.0, 0.0, id)
      : trade.Sell(InpLots, _Symbol, 0.0, 0.0, 0.0, id);
   if(!opened)
      PrintFormat("OPEN_FAILED id=%s direction=%d retcode=%u", id,
                  directions[next_signal], trade.ResultRetcode());
   else
      PrintFormat("OPEN_OK id=%s direction=%d time=%s price=%.5f", id,
                  directions[next_signal], TimeToString(bar_time, TIME_DATE | TIME_MINUTES),
                  trade.ResultPrice());
   next_signal++;
}

int OnInit()
{
   if(_Period != PERIOD_D1)
   {
      Print("HGE_SignalReplay must run on D1");
      return INIT_PARAMETERS_INCORRECT;
   }
   trade.SetExpertMagicNumber(InpMagic);
   trade.SetDeviationInPoints(InpMaxDeviationPoints);
   trade.SetTypeFillingBySymbol(_Symbol);
   if(!LoadSignals())
      return INIT_FAILED;
   return INIT_SUCCEEDED;
}

void OnTick()
{
   datetime current_bar = iTime(_Symbol, PERIOD_D1, 0);
   if(current_bar <= 0 || current_bar == last_bar_time)
      return;
   last_bar_time = current_bar;
   HandleNewBar(current_bar);
}

double OnTester()
{
   return TesterStatistics(STAT_PROFIT);
}
