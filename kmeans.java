import java.io.*;
import java.nio.channels.FileChannel;
import java.util.*;

public class Main {
    private double[][] feature;
    private int[] label;
    private String trainFileName;
    private String testFileName;
    private String predictFileName;
    private int theadNum = 3, timeReadLine = 2048, timeForTime;
    private StringBuffer sbf = new StringBuffer();
    private double[][] testFeature;
    private double[] d;
    private double[] p;

    public Main(String trainFileName, String testFileName, String predictFileName) {
        this.trainFileName = trainFileName;
        this.testFileName = testFileName;
        this.predictFileName = predictFileName;

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
    public void training() {
        loadTrainingData();
        int a = feature.length;
        int b = feature[0].length - 1;
        d = new double[b];
        p = new double[b];
        for(int i = 0; i < b; i++){
            double sum1 = 0;
            double sum2 = 0;
            int count = 0;
            for(int j = 0; j < a; j++){
                if(label[j] == 1){
                    sum1 += feature[j][i];
                    count++;
                }
                else {
                    sum2 += feature[j][i];
                }
            }
            d[i] = sum1/count;
            p[i] = sum2/(a-count);
        }
    }
    //训练数据读取
    public double[][] loadFilep(String fileName) {
        String result;
        try {
            FileInputStream fileInputStream = new FileInputStream(new File(fileName));
            FileChannel in = fileInputStream.getChannel();
            ByteArrayOutputStream outputStream = new ByteArrayOutputStream();
            byte[] buffer = new byte[1024];
            int len = 0;
            int k = 0;
            while(k < 3000){
                len = fileInputStream.read(buffer);
                outputStream.write(buffer,0,len);
                k++;
            }
            byte[] data = outputStream.toByteArray();
            fileInputStream.close();
            result = new String(data);
        } catch (IOException exception) {
            System.err.println(fileName + " File Not Found");
            return null;
        }
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
        for(int i = 0; i < a; i++){
            double sqr1 = 0;
            double sqr2 = 0;
            for(int j = 0; j < b; j++){
                sqr1 += (testFeature[i][j] - d[j]) * (testFeature[i][j] - d[j]);
                sqr2 += (testFeature[i][j] - p[j]) * (testFeature[i][j] - p[j]);
            }
            predictLabel[i] = sqr1 > sqr2 ? 0 : 1;

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

    public static void main(String[] args) {
		//填写相应文件的路径
        String trainFileName = "/data/train_data.txt";
        String testFileName = "/data/test_data.txt";
        String predictFileName = "/projects/student/result.txt";
        String answerFileName = "/projects/student/answer.txt";
        Main km = new Main(trainFileName, testFileName, predictFileName);
		
        Thread t1 =new Thread(new Runnable() {
            @Override
            public void run() {
                km.training();
            }
        });
		
        Thread t2 = new Thread(new Runnable() {
            @Override
            public void run() {
                km.loadpredictdata();
            }
        });
        try {
            t1.start();
            t2.start();
            t1.join();
            t2.join();
        }
        catch (Exception ex){
            System.out.println(ex.toString());
        }
        km.predict();
    }

}
