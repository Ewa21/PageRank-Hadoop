import com.google.common.collect.Lists;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.*;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.KeyValueTextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.text.DecimalFormat;
import java.text.NumberFormat;
import java.util.*;

public class PageRank extends Configured implements Tool {

    public static void main(String[] args) throws Exception {
        int res = ToolRunner.run(new Configuration(), new PageRank(), args);
        System.exit(res);
    }

    @Override
    public int run(String[] args) throws Exception {
        NumberFormat nf = new DecimalFormat("00");

        Configuration conf = this.getConf();
        int iterations = conf.getInt("K", 1);

        String inputPath;
        String outputPath = null;

        Job job = Job.getInstance(conf, "PageRank");
        specifyInitialJob(job, args[0], args[1]+"/00");
        boolean returnValue = job.waitForCompletion(true);

        for(int runs=0; runs <iterations && returnValue; runs++){
            job = Job.getInstance(conf, "PageRank");
            inputPath = args[1] + "/" + nf.format(runs);
            outputPath= args[1] + "/" + nf.format(runs+1);
            specifyInterativeJob(job, inputPath, outputPath);
            returnValue = job.waitForCompletion(true);
        }

        if(returnValue){
            job =Job.getInstance(conf, "PageRank");
            specifyFinalJob(job, outputPath, args[1]+"/Final");
            returnValue= job.waitForCompletion(true);
        }
        return returnValue ? 0 : 1;
    }

public void specifyInitialJob(Job job, String inputPath, String outputPath) throws IOException{
        job. setJarByClass(PageRank.class);
        job.setMapperClass(InitialMap.class);
        job.setMapOutputKeyClass(Text.class);
        job.setMapOutputValueClass(Text.class);

        job.setNumReduceTasks(1);
        FileInputFormat.addInputPath(job, new Path(inputPath));
        FileOutputFormat.setOutputPath(job, new Path(outputPath));
    }

    public void specifyInterativeJob(Job job, String inputPath, String outputPath) throws IOException{
        job. setJarByClass(PageRank.class);
        job.setMapperClass(PageRankMap.class);
        job.setMapOutputKeyClass(Text.class);
        job.setMapOutputValueClass(Text.class);

        job.setReducerClass(PageRankReducer.class);
        job.setInputFormatClass(KeyValueTextInputFormat.class);
        job.setNumReduceTasks(1);
        FileInputFormat.addInputPath(job, new Path(inputPath));
        FileOutputFormat.setOutputPath(job, new Path(outputPath));
    }

    public void specifyFinalJob(Job job, String inputPath, String outputPath) throws IOException{
        job. setJarByClass(PageRank.class);
        job.setMapperClass(SortingMap.class);

        job.setMapOutputKeyClass(FloatWritable.class);
        job.setMapOutputValueClass(Text.class);

        job.setInputFormatClass(KeyValueTextInputFormat.class);
        job.setNumReduceTasks(1);
        FileInputFormat.addInputPath(job, new Path(inputPath));
        FileOutputFormat.setOutputPath(job, new Path(outputPath));
    }

    public static class InitialMap extends Mapper<Object, Text, Text, Text> {
        private Text outKey = new Text();
        private Text outValue = new Text();
        private float initialRank;
        @Override
        protected void setup(Context context) throws IOException,InterruptedException {

            Configuration conf = context.getConfiguration();
            int N = conf.getInt("N", 5706070);
            this.initialRank = (float) 1.0/N;
        }

        @Override
        public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
			String line = value.toString();  //:A: B C D
            String[] lineArray = line.split(":");
            String page = lineArray[0]; //A
            outKey.set(page);
            String links = "";
            if(lineArray.length == 2) links = lineArray[1].trim();
            outValue.set(initialRank+" "+links);
            context.write(outKey, outValue);
        }
    }

    public static class SortingMap extends Mapper<Text, Text, FloatWritable, Text> {
        private FloatWritable pageRank = new FloatWritable();

        @Override
        public void map(Text key, Text value, Context context) throws IOException, InterruptedException {
            String[] rankAndOtherPages = value.toString().split(" ");
            pageRank.set(Float.valueOf(rankAndOtherPages[0]));
            context.write(pageRank, key);
        }
    }


    public static class PageRankMap extends Mapper<Text, Text, Text, Text> {

        @Override
        public void map(Text key, Text value, Context context) throws IOException, InterruptedException {

            String[] rankAndOtherPages = value.toString().split(" ");
            Float pageRank = Float.valueOf(rankAndOtherPages[0]);
            StringBuilder newValue = new StringBuilder("1");
            int pageCounter = rankAndOtherPages.length -1;
            float pageRankForLinks =pageRank / (float) pageCounter;
            for(int i=1; i<=pageCounter; i++){
                context.write(new Text(rankAndOtherPages[i]), new Text(String.valueOf(pageRankForLinks)));
                newValue.append(" ").append(rankAndOtherPages[i]);
            }
            context.write(key, new Text(newValue.toString()));
        }


    }

    public static class PageRankReducer extends Reducer<Text, Text, Text, Text> {
        Integer N;
        Float tax;
        Float beta;

        @Override
        protected void setup(Context context) throws IOException,InterruptedException {
            Configuration conf = context.getConfiguration();
            this.N = conf.getInt("N", 10);
            this.beta = conf.getFloat("B", 1);
            this.tax = (1-beta)/N;

        }

        @Override
        public void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
            Float pageRank=0.0f;
            StringBuilder links= new StringBuilder();
            for (Text val : values) {
                String[] rankAndOtherPages = val.toString().split(" ");
                if(rankAndOtherPages[0].equals("1")){
                    for(int i=1; i<rankAndOtherPages.length; i++){
                        links.append(" ").append(rankAndOtherPages[i]);
                    }
                }
                else{
                    pageRank+=Float.parseFloat(rankAndOtherPages[0]);
                }
            }
            context.write(key, new Text((beta*pageRank+tax)+links.toString()));
        }

    }
}
