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
		
		String temp = new InputHomogenization(text).transform();
		
		Collection<String> sentences = new ArrayList<String>();
		Reader reader = new StringReader(temp);
		temp = null;
		DocumentPreprocessor sentencesList = new DocumentPreprocessor(reader, DocType.Plain);
		for(List sentence:sentencesList) {
			temp = TextUtils.removeTags(sentence);
			sentences.add(temp);
			temp = null;
		}
		
		temp = StringUtils.join(sentences.toArray());
		sentences = null;
		
		//Tokenize the sentence and remove stop words
		TokenizerFactory tokenizerFactory = TextUtils.getTokenizerFactory(true);
		if(temp == null || temp.length()==0) return null;
		Tokenizer tokenizer =  tokenizerFactory.create(temp);
		List<String> tokens = tokenizer.getTokens();
		temp = null;
		
		int wordCount = 0;
		//Iterate over each token and get the vector of each token from word2vec
		for(int i =0;i<tokens.size();i++) {
			String token = tokens.get(i);
			if(wordVector.hasWord(token)) {
				double[] wordWeights = wordVector.getWordVector(token);
				wordList.add(wordWeights);
				wordWeights = null;
				wordCount++;
			}
			token = null;
		}
		tokens = null;

		double[] wordArray = wordList.get(0);
		for(int i=1;i<wordList.size();i++)
		{
			wordArray = ArrayUtils.addAll(wordArray, wordList.get(i));
		}
		
		
		double[] normalizedArray = new double[wordArray.length];
		for(int i=0;i<wordArray.length;i++) {
			normalizedArray[i] = wordArray[i]/wordCount;
		}
		wordArray = null;
		wordList = null;
		return normalizedArray;
	}

}
