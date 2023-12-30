package com.example.sleepppg;

import androidx.appcompat.app.AppCompatActivity;

import android.annotation.SuppressLint;
import android.graphics.Color;
import android.media.MediaPlayer;
import android.os.Bundle;
import android.content.res.AssetManager;
import android.util.Log;
import android.widget.TextView;
import android.widget.Toast;

import com.example.sleepppg.ml.SleepStageModelAuc02;
import com.github.mikephil.charting.charts.LineChart;
import com.github.mikephil.charting.components.XAxis;
import com.github.mikephil.charting.components.YAxis;
import com.github.mikephil.charting.data.Entry;
import com.github.mikephil.charting.data.LineData;
import com.github.mikephil.charting.data.LineDataSet;
import com.github.mikephil.charting.formatter.IndexAxisValueFormatter;
import com.github.mikephil.charting.formatter.ValueFormatter;
import com.github.mikephil.charting.interfaces.datasets.ILineDataSet;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

import java.util.ArrayList;

import java.util.Collections;

import java.util.Locale;


public class MainActivity extends AppCompatActivity {

    private SleepStageModelAuc02 model;
    private LineChart chart;
    private MediaPlayer deltaMediaPlayer;

    @SuppressLint("SetTextI18n")
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        deltaMediaPlayer = MediaPlayer.create(this, R.raw.delta_binaural_beat_fade);

        chart = findViewById(R.id.chart);
        // Setup the chart with empty data
        setupChart();
        chart.invalidate(); // refresh the chart

        if (!initModel()) {
            Toast.makeText(this, "Error initializing model", Toast.LENGTH_SHORT).show();
            return; // Stop further execution if model initialization fails
        }

        simulateNight();
//        displayFullNight();

    }



    private void displayFullNight() {
        try {
            // Replace with the actual number of data points per feature
            int dataPoints = 977;
            float[][] sleepData = new float[4][dataPoints];
            float[] timestamps = readValuesFromAsset("781756_time_feature.txt");
            sleepData[0] = readValuesFromAsset("781756_cosine_feature.txt");
            sleepData[1] = readValuesFromAsset("781756_count_feature.txt");
            sleepData[2] = readValuesFromAsset("781756_hr_feature.txt");
            sleepData[3] = timestamps;

            for (int i = 0; i < dataPoints; i++) {
                ByteBuffer inputBuffer = ByteBuffer.allocateDirect(4 * Float.BYTES).order(ByteOrder.nativeOrder());
                for (int j = 0; j < 4; j++) {
                    inputBuffer.putFloat(sleepData[j][i]);
                }
                inputBuffer.rewind();

                TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 4}, DataType.FLOAT32);
                inputFeature0.loadBuffer(inputBuffer);

                SleepStageModelAuc02.Outputs outputs = model.process(inputFeature0);
                TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();

                float[] prediction = outputFeature0.getFloatArray();

                float timestamp = timestamps[i];


                int sleepStageIndex = getSleepStageFromPrediction(prediction);
                final int finalSleepStageIndex = sleepStageIndex;

                runOnUiThread(() -> updateGraph(timestamp, finalSleepStageIndex));

            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
    private int nremSleepDuration = 0;
    private int nremSleepCounter = 0;
    private void simulateNight() {
        new Thread(() -> {
            try {
                int dataPoints = 977; // Replace with the actual number of data points
                float[][] sleepData = new float[4][dataPoints];
                float[] timestamps = readValuesFromAsset("781756_time_feature.txt");
                sleepData[0] = readValuesFromAsset("781756_cosine_feature.txt");
                sleepData[1] = readValuesFromAsset("781756_count_feature.txt");
                sleepData[2] = readValuesFromAsset("781756_hr_feature.txt");
                sleepData[3] = timestamps;

                for (int i = 0; i < dataPoints; i++) {
                    ByteBuffer inputBuffer = ByteBuffer.allocateDirect(4 * Float.BYTES).order(ByteOrder.nativeOrder());
                    for (int j = 0; j < 4; j++) {
                        inputBuffer.putFloat(sleepData[j][i]);
                    }
                    inputBuffer.rewind();

                    TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 4}, DataType.FLOAT32);
                    inputFeature0.loadBuffer(inputBuffer);

                    // Reinitialize the model for each prediction (if necessary)
                    model = SleepStageModelAuc02.newInstance(this);

                    SleepStageModelAuc02.Outputs outputs = model.process(inputFeature0);
                    TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();

                    float[] prediction = outputFeature0.getFloatArray();
                    float timestamp = timestamps[i];

                    int sleepStageIndex = getSleepStageFromPrediction(prediction);
                    final int finalSleepStageIndex = sleepStageIndex;

                    // Check if in NREM sleep (stage 2 or SWS)
                    if (finalSleepStageIndex == 2 || finalSleepStageIndex == 3) { // NREM sleep
                        nremSleepCounter++;
                        if (nremSleepCounter >= 10) { // Equivalent to 5 real-time minutes
                            runOnUiThread(() -> playDeltaBinauralBeat(finalSleepStageIndex));
                        }
                    } else {
                        nremSleepCounter = 0; // Reset if not in NREM sleep
                    }

                    runOnUiThread(() -> updateGraph(timestamp, finalSleepStageIndex));

                    Thread.sleep(1000); // Simulate a 1 second delay

                }
            } catch (IOException | InterruptedException e) {
                e.printStackTrace();
            }
        }).start();
    }


    private float currentVolume = 0.0f;
    private final float MAX_VOLUME_DB = 60.0f;

    private void playDeltaBinauralBeat(int sleepStageIndex) {
        if (sleepStageIndex == 2 || sleepStageIndex == 3) {
            if (!deltaMediaPlayer.isPlaying()) {
                adjustVolume();
                deltaMediaPlayer.start(); // Start playing the sound

            }
        } else {
            if (deltaMediaPlayer.isPlaying()) {
                stopAndResetMediaPlayer();
            }
        }
    }


    private void stopAndResetMediaPlayer() {
        if (deltaMediaPlayer.isPlaying()) {
            deltaMediaPlayer.pause(); // Pause when not in N2 or N3 stages
            deltaMediaPlayer.seekTo(0);
            resetVolume(); // Reset volume for next time
        }
        nremSleepCounter = 0; // Reset NREM sleep counter
    }


    private void adjustVolume() {
        if (currentVolume < MAX_VOLUME_DB) {
            currentVolume += 5; // Increase volume by 5%
            float volume = (float)(1 - (Math.log(MAX_VOLUME_DB - currentVolume) / Math.log(MAX_VOLUME_DB)));
            deltaMediaPlayer.setVolume(volume, volume);
        }
    }

    private void resetVolume() {
        currentVolume = 0.0f; // Reset volume to starting level
    }


    private int getSleepStageFromPrediction(float[] prediction) {
        int maxIndex = 0;
        float maxProbability = prediction[0];

        for (int i = 1; i < prediction.length; i++) {
            if (prediction[i] > maxProbability) {
                maxIndex = i;
                maxProbability = prediction[i];
            }
        }
        return maxIndex; // Map this index to your sleep stage labels as required
    }


    private String sleepStageIndexToLabel(int sleepStageIndex) {
        switch (sleepStageIndex) {
            case 0:
                return "Wake";
            case 1:
                return "N1";
            case 2:
                return "N2";
            case 3:
                return "N3";
            case 5:
                return "REM";
            default:
                return "Unknown";
        }
    }


    private void updateGraph(float timestamp, int sleepStageIndex) {
        // Adjust the index for the chart if it's above the skipped label
        int chartIndex = (sleepStageIndex >= 4) ? sleepStageIndex - 1 : sleepStageIndex;

        // Rest of the code remains the same
        String sleepStageLabel = sleepStageIndexToLabel(sleepStageIndex); // Map index to label
        Log.d("SleepStagePrediction", "Timestamp: " + timestamp + ", Prediction: " + sleepStageLabel);

        LineData data = chart.getData();
        if (data != null) {
            ILineDataSet set = data.getDataSetByIndex(0);
            if (set == null) {
                set = createSet();
                data.addDataSet(set);
            }

            // Use the adjusted index for plotting on the chart
            data.addEntry(new Entry(timestamp, chartIndex), 0);
            data.notifyDataChanged();

            chart.notifyDataSetChanged();
            chart.setVisibleXRangeMaximum(120);
            chart.moveViewToX(data.getEntryCount());

            set.addEntry(new Entry(timestamp, chartIndex));
        }
        chart.invalidate();
    }


    private float[] readValuesFromAsset(String filename) throws IOException {
        AssetManager am = getAssets();
        InputStream is = null;
        BufferedReader br = null;
        ArrayList<Float> valuesList = new ArrayList<>();
        try {
            is = am.open(filename);
            br = new BufferedReader(new InputStreamReader(is));
            String line;
            while ((line = br.readLine()) != null) {
                // Log each line read for debugging purposes
                Log.d("readValuesFromAsset", "Line read: " + line);
                valuesList.add(Float.parseFloat(line));
            }
        } catch (IOException | NumberFormatException e) {
            // Log the exception
            Log.e("readValuesFromAsset", "Error reading file", e);
            throw e;
        } finally {
            if (br != null) {
                br.close();
            }
            if (is != null) {
                is.close();
            }
        }
        float[] values = new float[valuesList.size()];
        for (int i = 0; i < values.length; i++) {
            values[i] = valuesList.get(i);
        }
        return values;
    }



    private void setupChart() {
        chart.getDescription().setEnabled(false);
        chart.setTouchEnabled(true);
        chart.setDragEnabled(true);
        chart.setScaleEnabled(true);
        chart.setDrawGridBackground(false);
        chart.setPinchZoom(true);
        chart.setBackgroundColor(Color.WHITE);

        LineData data = new LineData();
        data.setValueTextColor(Color.BLACK);
        chart.setData(data);

        configureChartAxis();
    }

    private void configureChartAxis() {
        XAxis xAxis = chart.getXAxis();
        xAxis.setPosition(XAxis.XAxisPosition.BOTTOM);
        xAxis.setDrawGridLines(false);
        xAxis.setValueFormatter(new ValueFormatter() {
            @Override
            public String getFormattedValue(float value) {
                // Convert the hour value into total minutes
                int totalMinutes = (int) (value * 60);
                int hours = totalMinutes / 60;
                int minutes = totalMinutes % 60;
                return String.format(Locale.getDefault(), "%02d:%02d", hours, minutes);
            }
        });

        YAxis leftAxis = chart.getAxisLeft();
        leftAxis.setDrawLabels(true);
        leftAxis.setValueFormatter(new IndexAxisValueFormatter(new String[]{"Wake", "N1", "N2", "N3", "REM"}));
        leftAxis.setAxisMinimum(0);
        leftAxis.setAxisMaximum(5);
        leftAxis.setGranularity(1f);
        chart.getAxisRight().setEnabled(false);
    }


    private LineDataSet createSet() {
        LineDataSet set = new LineDataSet(null, "Sleep Stages");
        set.setAxisDependency(YAxis.AxisDependency.LEFT);
        set.setLineWidth(2f);
        set.setHighlightEnabled(true);
        set.setDrawValues(false);
        set.setDrawCircles(false);
        set.setMode(LineDataSet.Mode.HORIZONTAL_BEZIER);
        set.setColors(getSleepStageColors());
        set.setValueTextColors(Collections.singletonList(Color.BLACK));
        return set;
    }

    private int[] getSleepStageColors() {
        return new int[]{Color.RED,    // Wake
                Color.BLUE,   // N1
                Color.GREEN,  // N2
                Color.YELLOW, // N3
                Color.MAGENTA // REM
        };
    }


    private boolean initModel() {
        try {
            model = SleepStageModelAuc02.newInstance(this);
            return true;
        } catch (IOException e) {
            e.printStackTrace();
            return false;
        }
    }

    @Override
    protected void onDestroy() {
        if (model != null) {
            model.close();
        }
        super.onDestroy();
    }
}
