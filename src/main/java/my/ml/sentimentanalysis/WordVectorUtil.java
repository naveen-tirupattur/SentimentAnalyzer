package my.ml.sentimentanalysis;

import java.util.Scanner;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import org.deeplearning4j.models.word2vec.Word2Vec;
import org.springframework.core.io.ClassPathResource;

public class WordVectorUtil {

	public static void main(String[] args) {
		ClassPathResource resource = new ClassPathResource("/wordVectorWithStopWords");
		Word2Vec wordVector;
		//try {
			Scanner inputScanner = new Scanner( System.in );
			String word = inputScanner.next();
			//Read the reviews and labels into a map
			Pattern p = Pattern.compile("\\s\\d\\t");
			Matcher m = p.matcher(word.toString());
			if(m.matches())
			{
				m.replaceAll("\t\\d\t");
			}
			System.out.println(word);
//			wordVector = SerializationUtils.readObject(resource.getInputStream());
//			System.out.println(wordVector.getWordVectorMatrix(word));
//			Collection<String> relatedWords = wordVector.wordsNearest(word, 10);
//					System.out.println(relatedWords);
			

//		} catch (IOException e) {
//			// TODO Auto-generated catch block
//			e.printStackTrace();
//		}
	}

}
