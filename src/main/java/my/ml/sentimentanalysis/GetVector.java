package my.ml.sentimentanalysis;

import java.io.Reader;
import java.io.StringReader;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

import org.apache.commons.lang.ArrayUtils;
import org.apache.commons.lang.StringUtils;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.text.inputsanitation.InputHomogenization;
import org.deeplearning4j.text.tokenization.tokenizer.Tokenizer;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;

import edu.stanford.nlp.process.DocumentPreprocessor;
import edu.stanford.nlp.process.DocumentPreprocessor.DocType;

public class GetVector {

	@SuppressWarnings({ "rawtypes", "unchecked" })
	public static double[] createVector(String text, Word2Vec wordVector, int numOfFeatures) {

		List<double[]> wordList = new ArrayList<double[]>();
		
		StringBuffer cleanedText = new StringBuffer();
		
		//Remove special characters
		cleanedText.append(new InputHomogenization(text).transform());
		
		Collection<String> sentences = new ArrayList<String>();
		Reader reader = new StringReader(cleanedText.toString());
		DocumentPreprocessor sentencesList = new DocumentPreprocessor(reader, DocType.Plain);
		for(List sentence:sentencesList) {
			sentences.add(TextUtils.removeTags(sentence));
		}
		
		//Clear the StringBuffer
		cleanedText.setLength(0);
		
		cleanedText.append(StringUtils.join(sentences.toArray()));
		
		sentences = null;
		//Tokenize the sentence and remove stop words
		TokenizerFactory tokenizerFactory = TextUtils.getTokenizerFactory(true);
		if(cleanedText == null || cleanedText.length()==0) return null;
		Tokenizer tokenizer =  tokenizerFactory.create(cleanedText.toString());
		List<String> tokens = tokenizer.getTokens();
		
		//Clear the StringBuffer
		cleanedText.setLength(0);
		
		int wordCount = 0;
		//Iterate over each token and get the vector of each token from word2vec
		for(int i =0;i<tokens.size();i++) {
			String token = tokens.get(i);
			if(wordVector.hasWord(token)) {
				wordList.add(wordVector.getWordVector(token));
				wordCount++;
			}
			token = null;
		}

		double[] wordArray = wordList.get(0);
		for(int i=1;i<wordList.size();i++)
		{
			wordArray = ArrayUtils.addAll(wordArray, wordList.get(i));
		}
		
		
		double[] normalizedArray = new double[wordArray.length];
		for(int i=0;i<wordArray.length;i++) {
			normalizedArray[i] = wordArray[i]/wordCount;
		}
		
		return normalizedArray;
	}

}
