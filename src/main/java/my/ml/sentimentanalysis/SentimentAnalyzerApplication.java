package my.ml.sentimentanalysis;

import java.io.IOException;
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
				Pair pair = new Pair(first,vector);
				first = null;
				second = null;
				TextUtils.writeFile("trainingData.csv", pair);
				pair = null;
				vector = null;
				count++;
			}			

		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

		long totalTimeInMilliSecs = System.currentTimeMillis() - startTime;
		System.out.println("Total time to finish in seconds: "+totalTimeInMilliSecs/(1000));

	}
}
