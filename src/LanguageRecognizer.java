import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;

public class LanguageRecognizer {

    public static HashMap<String, Double> nlMatrix = LangRegMapper.fillMatrix("/Users/Floortje/Downloads/HU-Letterfrequenties-main/Assignment2/src/textNL.txt");
    public static HashMap<String, Double> enMatrix = LangRegMapper.fillMatrix("/Users/Floortje/Downloads/HU-Letterfrequenties-main/Assignment2/src/textENG.txt");

    public static void main(String[] args) throws Exception {
        Job job = new Job();
        job.setJarByClass(LanguageRecognizer.class);
        
        FileInputFormat.addInputPath(job, new Path(args[0]));

        String pathname = args[1];

        FileOutputFormat.setOutputPath(job, new Path(pathname));

        job.setMapperClass(LangRegMapper.class);
        job.setReducerClass(RecognizeLanguageReducer.class);
        job.setInputFormatClass(TextInputFormat.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(IntWritable.class);

        job.waitForCompletion(true);
    }


    static class LangRegMapper extends Mapper<LongWritable, Text, Text, IntWritable> {
        private static HashMap<String, Double> createEmptyMatrix() {
            String alphabet = "abcdefghijklmnopqrstuvwxyz";
            Double score = 0.00;
            HashMap<String, Double> matrix = new HashMap<>();

            for (int i = 0; i < alphabet.length(); i++) {
                for (int j = 0; j < alphabet.length(); j++) {
                    char first = alphabet.charAt(i);
                    char second = alphabet.charAt(j);

                    StringBuffer pair = new StringBuffer().append(first).append(second);

                    matrix.put(pair.toString(), score);
                }
            }

            return matrix;
        }

        static ArrayList<String> cleanTextFile(String filePath) {
            BufferedReader reader;
            ArrayList<String> words = new ArrayList<>();

            try {
                reader = new BufferedReader(new FileReader(filePath));

                String line = reader.readLine();

                // Clean up file lines by removing symbols and adding them to words Array
                while (line != null) {
                    String clean = line.replaceAll("[^a-zA-Z\\s]", "");
                    String[] split = clean.split("\\s+");

                    for (String word : split) {
                        words.add(word);
                    }

                    line = reader.readLine();
                }
            } catch (IOException e) {
                e.printStackTrace();
            }

            return words;
        }

        private static HashMap<String, Double> fillMatrix(String filePath) {
            HashMap<String, Double> matrix = createEmptyMatrix();
            ArrayList<String> words = cleanTextFile(filePath);

            for (String word : words) {
                StringBuffer pair;
                HashMap<Character, Integer> countMap = new HashMap<>();
                HashMap<String, Integer> pairCountMap = new HashMap<>();

                // fill countMap with individual letter count for each word &
                // fill pairCountMap with a count for each letter pair
                for (int i = 0; i < word.length() - 1; i++) {
                    String cleanWord = word.toLowerCase();
                    pair = new StringBuffer().append(cleanWord.charAt(i)).append(cleanWord.charAt(i + 1));
                    char letter = cleanWord.charAt(i);

                    countMap.merge(letter, 1, Integer::sum);
                    pairCountMap.merge(pair.toString(), 1, Integer::sum);
                }

                // Calculate a score by the frequency of a letter pair and dividing it by the frequency of an individual letter &
                // adding that score to the matrix (averaging it if there is and existing score)
                for (String key : pairCountMap.keySet()) {
                    Character tracedLetter = key.charAt(0);
                    Integer tracedLetterCount = countMap.get(tracedLetter);
                    Integer pairCount = pairCountMap.get(key);
                    double score;

                    score = (float) pairCount / tracedLetterCount;


                    if (matrix.get(key) == 0.0) {
                        matrix.put(key, score);
                    } else {
                        // average is incorrect (should be divided by amount of appeared combinations
                        double average = (matrix.get(key) + score) / 2;
                        matrix.put(key, average);
                    }
                }

            }

            return matrix;
        }
    }

    static class RecognizeLanguageReducer extends Reducer<Text, IntWritable, Text, IntWritable> {
        public static void main(String[] args) {

            checkLanguage("C:/Users/Djurr/OneDrive/Documents/HBO-ICT/Jaar 3/Big Data/Assignment2/src/input.txt");
        }

        private static void checkLanguage(String filePath) {
            ArrayList<String> words = LangRegMapper.cleanTextFile(filePath);
            HashMap<String, Language> lineMap = new HashMap<>();
            BufferedReader reader;
            int countNL = 0;
            int countEN = 0;
            int countUND = 0;

            try {
                reader = new BufferedReader(new FileReader(filePath));

                String line = reader.readLine();

                while (line != null) {
                    HashMap<String, HashMap<String, Double>> wordScores = new HashMap<>();

                    for (String word : words) {
                        if (word.length() > 1) {
                            wordScores.put(word, checkWord(word));
                        }
                    }

                    line = line.replaceAll("[^a-zA-Z\\s]", "").toLowerCase();

                    lineMap.put(line, checkSentence(line, wordScores));

                    line = reader.readLine();
                }
            } catch (IOException e) {
                e.printStackTrace();
            }

//      count the amount of Dutch/English/Undefined sentences
            for (Language value : lineMap.values()) {
                if (value == Language.Nederlands) {
                    countNL++;
                } else if (value == Language.Engels) {
                    countEN++;
                } else {
                    countUND++;
                }
            }

            for (String key : lineMap.keySet()) {
                System.out.println(key + " || " + lineMap.get(key));
            }

            System.out.println("Zinnen Engels: " + countEN);
            System.out.println("Zinnen Nederlands: " + countNL);
            System.out.println("Zinnen Onbekend: " + countUND);
        }

        //  get the total score for all letter pairs in the word based on the English and Dutch matrix
        private static HashMap<String, Double> checkWord(String word) {
            StringBuffer pair;
            HashMap<String, Double> scoreMap = new HashMap<>();
            String cleanWord = word.toLowerCase();


            for (int i = 0; i < word.length() - 1; i++) {
                double nlPairScore;
                double enPairScore;

                pair = new StringBuffer().append(cleanWord.charAt(i)).append(cleanWord.charAt(i + 1));

                nlPairScore = LanguageRecognizer.nlMatrix.get(pair.toString());
                enPairScore = LanguageRecognizer.enMatrix.get(pair.toString());

                if (scoreMap.get(cleanWord) == null) {
                    scoreMap.put("nl", nlPairScore);
                } else {
                    scoreMap.put("nl", scoreMap.get(cleanWord) + nlPairScore);
                }

                if (scoreMap.get(cleanWord) == null) {
                    scoreMap.put("en", enPairScore);
                } else {
                    scoreMap.put("en", scoreMap.get(cleanWord) + enPairScore);
                }
            }

            return scoreMap;
        }

        //  Add up the score for each word in the sentence and return the language which score was the highest
        private static Language checkSentence(String sentence, HashMap<String, HashMap<String, Double>> wordMap) {
            String[] split = sentence.split("\\s+");
            double scoreNL = 0.00;
            double scoreEN = 0.00;

            for (String word : split) {
                if (wordMap.get(word) != null) {
                    scoreNL += wordMap.get(word).get("nl");
                    scoreEN += wordMap.get(word).get("en");
                } else {
                    scoreNL = 0.00;
                    scoreEN = 0.00;
                }
            }

            if (scoreNL > scoreEN) {
                return Language.Nederlands;
            } else if (scoreEN > scoreNL) {
                return Language.Engels;
            } else {
                return Language.Undefined;
            }

        }

        enum Language {
            Nederlands,
            Engels,
            Undefined
        }
    }
}


