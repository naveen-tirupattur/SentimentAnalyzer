package my.ml.sentimentanalysis;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.util.SerializationUtils;
import org.springframework.core.io.ClassPathResource;


public class SentimentAnalyzerApplication {

	@SuppressWarnings({ "rawtypes","unchecked" })
	public static void main(String args[]) {

		long startTime = System.currentTimeMillis();

		int numOfFeatures = 300;

		//generate the wordvector from the data
		//GenerateWordVector.createWordVector("/labeledTrainData.tsv",numOfFeatures);

		List<Pair<String,double[]>> trainingData = new ArrayList<Pair<String,double[]>>();

		//read the wordvector
		ClassPathResource resource = new ClassPathResource("/wordVectorWithStopWords");
		Word2Vec wordVector;
		try {
			wordVector = SerializationUtils.readObject(resource.getInputStream());
			//			Collection<String> relatedWords = wordVector.wordsNearest("cool", 10);
			//		System.out.println(relatedWords);

			//Read the dataset from file
			Map<String, Pair<String,String>> dataMap = TextUtils.getDataMap("/labeledTrainData.tsv");
			System.out.println("Dataset size: "+dataMap.size());
			int count = 0;
			String first = "", second = "";
			for(Pair p: dataMap.values() ) {
				first = p.getFirst().toString();
				second = p.getSecond().toString();
				if(count%1000 == 0) System.out.println("Processed "+count+" records");
				double[] vector = GetVector.createVector(second, wordVector, numOfFeatures);
				first = null;
				second = null;
				trainingData.add(new Pair(first,vector));
				vector = null;
				count++;
				
			}

			TextUtils.writeFile("trainingData.csv", trainingData);

		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

		long totalTimeInMilliSecs = System.currentTimeMillis() - startTime;
		System.out.println("Total time to finish in seconds: "+totalTimeInMilliSecs/(1000));

	}
}
