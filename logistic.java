import java.io.*;
import java.nio.channels.FileChannel;
import java.util.*;
import java.util.concurrent.ThreadLocalRandom;

public class Main {
    private double[][] feature;
    private int[] label;
    private double stepLength;
    private int maxStep;
    private double initWeight;
    private double[] weights;
    private String trainFileName;
    private String testFileName;
    private String predictFileName;
    private int theadNum = 3, timeReadLine = 2048, timeForTime;
    private StringBuffer sbf = new StringBuffer();
    private StringBuffer sbf1 = new StringBuffer();
    private double[][] testFeature;

    public Main(String trainFileName, String testFileName, String predictFileName) {
        this.trainFileName = trainFileName;
        this.testFileName = testFileName;
        this.predictFileName = predictFileName;

        this.stepLength = 0.01;
        this.maxStep = 10000;
        this.initWeight = 0.01;
    }

    private void loadTrainingData() {
        double[][] matrix = loadFilep(trainFileName);
        int a = matrix.length;
        int b= matrix[0].length;
        feature = new double[a][b];
        label = new int[a];
        for (int i = 0; i < a; i++) {
            for (int j = 0; j < b - 1; j++) {
                feature[i][j] = matrix[i][j];
            }
            label[i] = (int) matrix[i][b - 1];
        }
    }

    private void initWeightMatrix() {
        int parnum = feature[0].length;
        double[] weights = new double[parnum];
        for(int i = 0; i < parnum; i++){
            weights[i] = initWeight;
        }
        this.weights = weights;
    }

    private double getPredictLabel(int i) {
        double predictSum = 0;
        int a = feature[0].length;
        for (int j = 0; j < a; j++) {
            predictSum += feature[i][j] * weights[j];
        }
        double p = sigmoid(predictSum);
        return p;
    }

    public void training() {
        loadTrainingData();
        initWeightMatrix();
        int a = feature.length;
        int b= feature[0].length;
        long end = System.currentTimeMillis();
        for (int i = 0; i < maxStep; i++) {
            int x = ThreadLocalRandom.current().nextInt(a);
            double p = getPredictLabel(x);
            for (int j = 0; j < b; j++) {
                weights[j] += stepLength * (feature[x][j] * (label[x] - p));
            }
        }
        System.out.println("训练过程："+(System.currentTimeMillis() - end)*1.0/1000);
        this.feature = null;
        this.label = null;
    }
    //训练数据读取
    public double[][] loadFilep(String fileName) {
        long start = System.currentTimeMillis();
        read1(new File(fileName), 0);
        System.out.println("训练数据读取时间："+(System.currentTimeMillis()-start)*1.0/1000);
        String result = new String(sbf1);
        long end = System.currentTimeMillis();
        List<List<Double>> listArr = new ArrayList<>();
        String temp[] = result.split("\n");
        int a = temp.length;
        for(int i = 0; i < a; i++) {
            List<Double> list = new ArrayList<>();
            String item[] = temp[i].split(",");
            for (int j = 0; j < item.length; j++) {
                list.add(Double.parseDouble(item[j]));
            }
            listArr.add(list);
        }
        double[][] matrix = new double[listArr.size()][listArr.get(0).size()];
        for (int i = 0; i < listArr.size(); i++) {
            for (int j = 0; j < listArr.get(i).size(); j++) {
                matrix[i][j] = listArr.get(i).get(j);
            }
        }
        System.out.println("字符分割时间："+(System.currentTimeMillis()-end)*1.0/1000);
        return matrix;
    }

    //预测数据读取
    public double[][] loadFile(String fileName) {
        read(new File(fileName), 0);
        String result = new String(sbf);
        List<List<Double>> listArr = new ArrayList<>();
        String temp[] = result.split("\n");
        for(int i = 0; i < temp.length; i++) {
            List<Double> list = new ArrayList<>();
            String item[] = temp[i].split(",");
            for (int j = 0; j < item.length; j++) {
                list.add(Double.parseDouble(item[j]));
            }
            listArr.add(list);
        }
        double[][] matrix = new double[listArr.size()][listArr.get(0).size()];
        for (int i = 0; i < listArr.size(); i++) {
            for (int j = 0; j < listArr.get(i).size(); j++) {
                matrix[i][j] = listArr.get(i).get(j);
            }
        }
        return matrix;
    }
    //多线程读取....................................
    public void read(File file, int start){
        List<ReadItem> list = new ArrayList<Main.ReadItem>();
        long le = file.length();
        timeForTime = (int)le/(theadNum*timeReadLine) + 1;
        for(int i = 0; i < theadNum; i++){
            list.add(new ReadItem(file, start + i*timeReadLine*timeForTime));	//创建多个子线程
            list.get(list.size()-1).start();
        }

        for(int i = 0; i < list.size(); i++){
            try {
                list.get(i).join();
                sbf.append(list.get(i).getSb());
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
    }
    public void read1(File file, int start){
        List<ReadItem> list = new ArrayList<Main.ReadItem>();
        long le = file.length();
        timeForTime = (int)le/(theadNum*timeReadLine) + 1;
        for(int i = 0; i < theadNum; i++){
            list.add(new ReadItem(file, start + i*timeReadLine*timeForTime));	//创建多个子线程
            list.get(list.size()-1).start();
        }

        for(int i = 0; i < list.size(); i++){
            try {
                list.get(i).join();
                sbf1.append(list.get(i).getSb());
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
    }

    public class ReadItem extends Thread{
        private BufferedReader reader;
        private int start, lastNum;
        private StringBuffer sb;

        public ReadItem(File file, int start) {
            try {
                this.reader = new BufferedReader(new InputStreamReader(new FileInputStream(file)));
            } catch (Exception e) {
                e.printStackTrace();
            }
            this.start = start;
            this.sb = new StringBuffer();
        }

        @Override
        public void run() {
            char[] buf = new char[timeReadLine];
            try {
                reader.skip(this.start);
                for (int i = 0; i < timeForTime && (lastNum = reader.read(buf)) != -1; i++ ) {
                    sb.append(new String(buf,0,lastNum));
                }
            } catch (Exception e) {
                e.printStackTrace();
            }
        }

        public int getLastNum() {
            return lastNum;
        }

        public StringBuffer getSb() {
            return sb;
        }
    }

    //读取测试文件
    public void loadpredictdata(){
        testFeature = loadFile(testFileName);
    }


    //预测及保存
    public void predict() {
        int[] predictLabel = new int[testFeature.length];
        int a = testFeature.length;
        int b =testFeature[0].length;
        for (int i = 0; i < a; i++) {
            double sum = 0;
            for (int j = 0; j < b; j++) {
                sum += testFeature[i][j] * weights[j];
            }
            predictLabel[i] = sigmoid(sum) > 0.5 ? 1 : 0;
        }
        int c = predictLabel.length;
        try {
            BufferedWriter out = new BufferedWriter(new FileWriter(predictFileName));
            for (int i = 0; i < c; i++) {
                out.write(predictLabel[i] + "\n");
            }
            out.close();
        } catch (IOException exception) {
            System.err.println(exception.getMessage());
        }
    }

    private double sigmoid(double x) {
        return 1.0d / (1.0d + Math.exp(-x));
    }

    public static void main(String[] args) {
        //文件路径
        String trainFileName = "E:/code_craft2020/train_data.txt";
        String testFileName = "E:/code_craft2020/test_data.txt";
        String predictFileName = "E:/code_craft2020/result.txt";
        String answerFileName = "E:/code_craft2020/answer.txt";
        long start = System.currentTimeMillis();
        Main lr = new Main(trainFileName, testFileName, predictFileName);
        lr.training();
        lr.loadpredictdata();
        lr.predict();
        System.out.println((System.currentTimeMillis() - start)*1.0/1000);
    }

}
