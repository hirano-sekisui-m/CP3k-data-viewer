import numpy as np

# Spline might be monotonic cubic spline or similar.
# But user said "まずは、目的①の解決を目指します。"
# Which is "試薬パラメーターの至適条件を決定するため、測光ポイント(測光開始(秒)と終了(秒))の全組合せでキャリブレーターの処理値を求め、これを使用して検量線を作成します。そのキャリブレーションデータを用いたと仮定して、変更後の測光ポイントで一般検体の処理値も算出し、検体測定値まで求めます。その上で、別の測光ポイントを選択した際との吸光度および測定値がどのように変化するかの比較を行いたいです。"

# We need to build a UI in streamlit `main.py` that allows:
# 1. Select a generic method / item (e.g., TAT207).
# 2. Input/configure calibrator info (concentrations, calibrator requests, curve fit mode like Polyline or Spline).
# 3. Select alternative photometer points (start & end).
# 4. Show the new calibration curve, calculate sample concentrations based on the new points, and compare with original sample concentrations.
