;layer_height = 0.4
;line_width = 0.4
;layer_count = 1
;mesh = single_45_top
G21 ;start of the code
G1 Z15 F300
G28 X0 Y0 ;Home
G92 X0 Y0 ;Consider this as current
G0 X50 Y50 F3000 ;Go-to Offset
G92 X0 Y0 ;Reset

G0 F3600 X0.000 Y0.000 Z0.300 ;Go to start position
M7
G4 P150


G1 X9.800 Y9.800 Z0.300
G1 X9.800 Y9.800
G1 X9.800 Y10.200
G1 X35.367 Y10.200
G1 X35.367 Y9.800
G1 X9.800 Y9.800
G1 X9.400 Y9.400
G1 X9.400 Y10.600
G1 X35.767 Y10.600
G1 X35.767 Y9.400
G1 X9.400 Y9.400
G1 X9.000 Y9.000
G1 X9.000 Y11.000
G1 X36.167 Y11.000
G1 X36.167 Y9.000
G1 X9.000 Y9.000
G1 X8.600 Y8.600
G1 X8.600 Y11.400
G1 X36.567 Y11.400
G1 X36.567 Y8.600
G1 X8.600 Y8.600
G1 X8.200 Y8.200
G1 X8.200 Y11.800
G1 X36.967 Y11.800
G1 X36.967 Y8.200
G1 X8.200 Y8.200
G1 X7.800 Y7.800
G1 X7.800 Y12.200
G1 X37.367 Y12.200
G1 X37.367 Y7.800
G1 X7.800 Y7.800
G1 X7.400 Y7.400
G1 X7.400 Y12.600
G1 X37.767 Y12.600
G1 X37.767 Y7.400
G1 X7.400 Y7.400
G1 X7.000 Y7.000
G1 X7.000 Y13.000
G1 X38.167 Y13.000
G1 X38.167 Y7.000
G1 X7.000 Y7.000
G1 X6.600 Y6.600
G1 X6.600 Y13.400
G1 X38.567 Y13.400
G1 X38.567 Y6.600
G1 X6.600 Y6.600
G1 X6.200 Y6.200
G1 X6.200 Y13.800
G1 X38.967 Y13.800
G1 X38.967 Y6.200
G1 X6.200 Y6.200
G1 X5.800 Y5.800
G1 X5.800 Y14.200
G1 X39.367 Y14.200
G1 X39.367 Y5.800
G1 X5.800 Y5.800
G1 X5.400 Y5.400
G1 X5.400 Y14.600
G1 X39.767 Y14.600
G1 X39.767 Y5.400
G1 X5.400 Y5.400
G1 X5.000 Y5.000
G1 X5.000 Y15.000
G1 X40.167 Y15.000
G1 X40.167 Y5.000
G1 X5.000 Y5.000
G1 X4.600 Y4.600
G1 X4.600 Y15.400
G1 X40.567 Y15.400
G1 X40.567 Y4.600
G1 X4.600 Y4.600
G1 X4.200 Y4.200
G1 X4.200 Y15.800
G1 X40.967 Y15.800
G1 X40.967 Y4.200
G1 X4.200 Y4.200
G1 X3.800 Y3.800
G1 X3.800 Y16.200
G1 X41.367 Y16.200
G1 X41.367 Y3.800
G1 X3.800 Y3.800
G1 X3.400 Y3.400
G1 X3.400 Y16.600
G1 X41.767 Y16.600
G1 X41.767 Y3.400
G1 X3.400 Y3.400
G1 X3.000 Y3.000
G1 X3.000 Y17.000
G1 X42.167 Y17.000
G1 X42.167 Y3.000
G1 X3.000 Y3.000
G1 X2.600 Y2.600
G1 X2.600 Y17.400
G1 X42.567 Y17.400
G1 X42.567 Y2.600
G1 X2.600 Y2.600
G1 X2.200 Y2.200
G1 X2.200 Y17.800
G1 X42.967 Y17.800
G1 X42.967 Y2.200
G1 X2.200 Y2.200
G1 X1.800 Y1.800
G1 X1.800 Y18.200
G1 X43.367 Y18.200
G1 X43.367 Y1.800
G1 X1.800 Y1.800
G1 X1.400 Y1.400
G1 X1.400 Y18.600
G1 X43.767 Y18.600
G1 X43.767 Y1.400
G1 X1.400 Y1.400
G1 X1.000 Y1.000
G1 X1.000 Y19.000
G1 X44.167 Y19.000
G1 X44.167 Y1.000
G1 X1.000 Y1.000
G1 X0.600 Y0.600
G1 X0.600 Y19.400
G1 X44.567 Y19.400
G1 X44.567 Y0.600
G1 X0.600 Y0.600
G1 X0.200 Y0.200
G1 X0.200 Y19.800
G1 X44.967 Y19.800
G1 X44.967 Y0.200
G1 X0.200 Y0.200
G1 X9.800 Y9.800 Z0.700
G1 X9.800 Y9.800
G1 X9.800 Y10.200
G1 X35.367 Y10.200
G1 X35.367 Y9.800
G1 X9.800 Y9.800
G1 X9.400 Y9.400
G1 X9.400 Y10.600
G1 X35.767 Y10.600
G1 X35.767 Y9.400
G1 X9.400 Y9.400
G1 X9.000 Y9.000
G1 X9.000 Y11.000
G1 X36.167 Y11.000
G1 X36.167 Y9.000
G1 X9.000 Y9.000
G1 X8.600 Y8.600
G1 X8.600 Y11.400
G1 X36.567 Y11.400
G1 X36.567 Y8.600
G1 X8.600 Y8.600
G1 X8.200 Y8.200
G1 X8.200 Y11.800
G1 X36.967 Y11.800
G1 X36.967 Y8.200
G1 X8.200 Y8.200
G1 X7.800 Y7.800
G1 X7.800 Y12.200
G1 X37.367 Y12.200
G1 X37.367 Y7.800
G1 X7.800 Y7.800
G1 X7.400 Y7.400
G1 X7.400 Y12.600
G1 X37.767 Y12.600
G1 X37.767 Y7.400
G1 X7.400 Y7.400
G1 X7.000 Y7.000
G1 X7.000 Y13.000
G1 X38.167 Y13.000
G1 X38.167 Y7.000
G1 X7.000 Y7.000
G1 X6.600 Y6.600
G1 X6.600 Y13.400
G1 X38.567 Y13.400
G1 X38.567 Y6.600
G1 X6.600 Y6.600
G1 X6.200 Y6.200
G1 X6.200 Y13.800
G1 X38.967 Y13.800
G1 X38.967 Y6.200
G1 X6.200 Y6.200
G1 X5.800 Y5.800
G1 X5.800 Y14.200
G1 X39.367 Y14.200
G1 X39.367 Y5.800
G1 X5.800 Y5.800
G1 X5.400 Y5.400
G1 X5.400 Y14.600
G1 X39.767 Y14.600
G1 X39.767 Y5.400
G1 X5.400 Y5.400
G1 X5.000 Y5.000
G1 X5.000 Y15.000
G1 X40.167 Y15.000
G1 X40.167 Y5.000
G1 X5.000 Y5.000
G1 X4.600 Y4.600
G1 X4.600 Y15.400
G1 X40.567 Y15.400
G1 X40.567 Y4.600
G1 X4.600 Y4.600
G1 X4.200 Y4.200
G1 X4.200 Y15.800
G1 X40.967 Y15.800
G1 X40.967 Y4.200
G1 X4.200 Y4.200
G1 X3.800 Y3.800
G1 X3.800 Y16.200
G1 X41.367 Y16.200
G1 X41.367 Y3.800
G1 X3.800 Y3.800
G1 X3.400 Y3.400
G1 X3.400 Y16.600
G1 X41.767 Y16.600
G1 X41.767 Y3.400
G1 X3.400 Y3.400
G1 X3.000 Y3.000
G1 X3.000 Y17.000
G1 X42.167 Y17.000
G1 X42.167 Y3.000
G1 X3.000 Y3.000
G1 X2.600 Y2.600
G1 X2.600 Y17.400
G1 X42.567 Y17.400
G1 X42.567 Y2.600
G1 X2.600 Y2.600
G1 X2.200 Y2.200
G1 X2.200 Y17.800
G1 X42.967 Y17.800
G1 X42.967 Y2.200
G1 X2.200 Y2.200
G1 X1.800 Y1.800
G1 X1.800 Y18.200
G1 X43.367 Y18.200
G1 X43.367 Y1.800
G1 X1.800 Y1.800
G1 X1.400 Y1.400
G1 X1.400 Y18.600
G1 X43.767 Y18.600
G1 X43.767 Y1.400
G1 X1.400 Y1.400
G1 X1.000 Y1.000
G1 X1.000 Y19.000
G1 X44.167 Y19.000
G1 X44.167 Y1.000
G1 X1.000 Y1.000
G1 X0.600 Y0.600
G1 X0.600 Y19.400
G1 X44.567 Y19.400
G1 X44.567 Y0.600
G1 X0.600 Y0.600
G1 X0.200 Y0.200
G1 X0.200 Y19.800
G1 X44.967 Y19.800
G1 X44.967 Y0.200
G1 X0.200 Y0.200

;Finish
M9
G1 Z10.000
G28 X-100 Y-100;Home