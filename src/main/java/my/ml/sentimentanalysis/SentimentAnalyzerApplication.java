package my.ml.sentimentanalysis;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.util.SerializationUtils;


public class SentimentAnalyzerApplication {

	@SuppressWarnings({ "rawtypes","unchecked" })
	public static void main(String args[]) {

		long startTime = System.currentTimeMillis();

		int numOfFeatures = 300;

		//generate the wordvector from the data
		//GenerateWordVector.createWordVector("/labeledTrainData.tsv",numOfFeatures);

		List<Pair<String,String>> trainingData = new ArrayList<Pair<String,String>>();

		//read the wordvector
		Word2Vec wordVector = SerializationUtils.readObject(new File("wordVectorWithStopWords"));

		//		Collection<String> relatedWords = wordVector.wordsNearest("cool", 10);
		//		System.out.println(relatedWords);

		//Read the dataset from file
		Map<String, Pair<String,String>> dataMap = TextUtils.getDataMap("/labeledTrainData.tsv");
		System.out.println("Dataset size: "+dataMap.size());
		int count = 0;
		for(Pair p: dataMap.values() ) {
			if(count%1000 == 0) System.out.println("Processed "+count+" records");
			double[] vector = GetVector.createVector(p.getSecond().toString(), wordVector, numOfFeatures);
			trainingData.add(new Pair(p.getFirst().toString(),Arrays.toString(vector)));
			count++;
		}

		TextUtils.writeFile("trainingData.csv", trainingData);
		long totalTimeInMilliSecs = System.currentTimeMillis() - startTime;

		System.out.println("Total time to finish in seconds: "+totalTimeInMilliSecs/(1000));

	}
}
