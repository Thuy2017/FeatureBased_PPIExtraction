
import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Properties;
import java.util.Random;

import javax.xml.parsers.DocumentBuilderFactory;
import javax.xml.xpath.XPathConstants;
import javax.xml.xpath.XPathExpression;
import javax.xml.xpath.XPathExpressionException;
import javax.xml.xpath.XPathFactory;

import org.w3c.dom.Document;
import org.w3c.dom.NamedNodeMap;
import org.w3c.dom.Node;
import org.w3c.dom.NodeList;

import edu.stanford.nlp.ling.CoreAnnotations.LemmaAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.SentencesAnnotation;
//import edu.stanford.nlp.ling.CoreAnnotations.NamedEntityTagAnnotation;
//import edu.stanford.nlp.ling.CoreAnnotations.PartOfSpeechAnnotation;
//import edu.stanford.nlp.ling.CoreAnnotations.SentencesAnnotation;
//import edu.stanford.nlp.ling.CoreAnnotations.TextAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.TokensAnnotation;
//import edu.stanford.nlp.pipeline.Annotation;
//import edu.stanford.nlp.pipeline.StanfordCoreNLP;
//import edu.stanford.nlp.util.CoreMap;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.parser.lexparser.LexicalizedParser;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.trees.TypedDependency;
//import edu.stanford.nlp.trees.TreePrint;
import edu.stanford.nlp.util.CoreMap;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.SMO;
//import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class PPIextraction {
public static String corpus = "BioInfer";

	public static class Entity{      //Entityクラス　HashMap登録に使用
		public Entity(String origId,String text,int charoffsetF,int charoffsetR){
			this.origId = origId;
			this.text = text;
			this.charoffsetF = charoffsetF;
			this.charoffsetR = charoffsetR;
		}
		String origId;
		String text;
		int charoffsetF;
		int charoffsetR;
	}

	public static class token{      //tokenクラス　HashMap登録に使用
		public token(String POS, String text,String id) {
			this.POS = POS;
			this.text = text;
			this.id = id;
		}
		String POS;
		String text;
		String id;
	}

	private static void WriteRelword(PrintWriter pw,String relword){
		pw.print("@attribute interactionword {");

		try{
			FileReader in = new FileReader(relword);
			BufferedReader br = new BufferedReader(in);
			String line;
			while ((line = br.readLine()) != null) {
				pw.print(line);
				pw.print(",");
			}
			pw.println("norelword}");
			br.close();
			in.close();
		}catch(Exception e){
			System.out.println(e);
		}
	}


	private static void write(String filename,String relword,String lemma_relword){   //学習用のarffファイル作成　Feature一覧を書き込み
		File outputFile = new File(filename);
		FileOutputStream fos;
		try{
			fos = new FileOutputStream(outputFile);
			OutputStreamWriter osw = new OutputStreamWriter(fos);
			PrintWriter pw = new PrintWriter(osw);

			pw.println("@relation interact");
			pw.println("");
			WriteRelword(pw,lemma_relword);
			pw.println("@attribute D1 real");
			pw.println("@attribute D2 real");
			pw.println("@attribute D3 real");
			pw.println("@attribute P_posision1 real");
			pw.println("@attribute P_position2 real");
			pw.println("@attribute K_position real");
			pw.println("@attribute order {avb,vab,abv}");
		//	pw.println("@attribute NumberOfInteractors {high,low}");
			pw.println("@attribute NumberOfInteractors real"); //number of relation-words
		//	pw.println("@attribute comma {nn,yn,ny,yy}");
			pw.println("@attribute not {true,false}");
		//	pw.println("@attribute breaker {true,false}");
		//	pw.println("@attribute conditional {true,false}");
			pw.println("@attribute prep {with,of,by,multiple,to,between,via,through,in,for,on,within,during,from,without,at,under,among,after,NA}");
		//	pw.println("@attribute which {true,false}");
		//	pw.println("@attribute but {true,false}");
			pw.println("@attribute H1 real");
			pw.println("@attribute H2 real");
			pw.println("@attribute H3 real");

		/*  pw.println("@attribute POS1 real");
			pw.println("@attribute POS2 real");
			pw.println("@attribute POS3 real");
		*/
			pw.println("@attribute overlapP {true,false}");
			pw.println("@attribute overlapN {true,false}");
			pw.println("@attribute overlap2 {true,false}");
			pw.println("@attribute overlap3 {true,false}");
			pw.println("@attribute overlapF {true,false}");
			pw.println("@attribute overlapR {true,false}");

			pw.println("@attribute pattern_PVP {true,false}");
			pw.println("@attribute pattern_PVbyP {true,false}");
			pw.println("@attribute pattern_VofPbyP {true,false}");
			pw.println("@attribute pattern_VofPtoP {true,false}");
			pw.println("@attribute pattern_NofPbyP {true,false}");
			pw.println("@attribute pattern_NofPwithP {true,false}");
			pw.println("@attribute pattern_NbetweenPandP {true,false}");
			pw.println("@attribute pattern_combetPandP {true,false}");
			pw.println("@attribute pattern_comofPandP {true,false}");
			pw.println("@attribute pattern_PformcomwithP {true,false}");
			pw.println("@attribute pattern_PPN {true,false}");
			pw.println("@attribute pattern_PdependP {true,false}");
			pw.println("@attribute pattern_between {true,false}");


			pw.println("@attribute pattern_contiguous {true,false}");
			pw.println("@attribute pattern_para {true,false}");

		/*	pw.println("@attribute pattern_bind {true,false}");
			pw.println("@attribute pattern_interact {true,false}");
			pw.println("@attribute pattern_known {true,false}");
			pw.println("@attribute pattern_regulate {true,false}");
			pw.println("@attribute pattern_induce {true,false}");
			pw.println("@attribute pattern_stimulate {true,false}");
			pw.println("@attribute pattern_associate {true,false}");
		*/
			pw.println("@attribute Dep_KP1 real");
			pw.println("@attribute negative_KP1 real");
			pw.println("@attribute positive_KP1 real");

			pw.println("@attribute Dep_KP2 real");
			pw.println("@attribute negative_KP2 real");
			pw.println("@attribute positive_KP2 real");

			pw.println("@attribute Dep_P1P2 real");
			pw.println("@attribute negative_P1P2 real");
			pw.println("@attribute positive_P1P2 real");

			pw.println("@attribute Dep_P1Root real");
			pw.println("@attribute Dep_P2Root real");
			pw.println("@attribute Dep_KRoot real");

			pw.println("@attribute interaction {True,False}");
			pw.println("");
			pw.println("@data");

			pw.close();
		}catch(FileNotFoundException e){
			e.printStackTrace();
		}
	}

	private static HashSet<String> WordsSet(String filename,HashSet<String> hs){     //RelationWordsをハッシュセットに登録
		try{
			FileReader in = new FileReader(filename);
			BufferedReader br = new BufferedReader(in);
			String line;
			while ((line = br.readLine()) != null) {
				line = line.toUpperCase();
				hs.add(line);
			}
			br.close();
			in.close();
			return (hs);
		}catch(Exception e){
			System.out.println(e);
			return(hs);
		}
	}

	private static void addwriteFirst(String filename,String str){
		File outputFile = new File(filename);
		FileOutputStream fos;
		try{
			fos = new FileOutputStream(outputFile,true);
			OutputStreamWriter osw = new OutputStreamWriter(fos);
			PrintWriter pw = new PrintWriter(osw);

			pw.print(str);
			pw.close();
		}catch(IOException e){
			System.out.println(e);
		}
	}

	private static void addwrite(String filename,String str){
		File outputFile = new File(filename);
		FileOutputStream fos;
		try{
			fos = new FileOutputStream(outputFile,true);
			OutputStreamWriter osw = new OutputStreamWriter(fos);
			PrintWriter pw = new PrintWriter(osw);
			pw.print(","+str);
			pw.close();
		}catch(IOException e){
			System.out.println(e);
		}
	}

	private static void addwrite(String filename,String[] str){
		File outputFile = new File(filename);
		FileOutputStream fos;
		int num = str.length;
		//System.out.println(num);
		try{
			fos = new FileOutputStream(outputFile,true);
			OutputStreamWriter osw = new OutputStreamWriter(fos);
			PrintWriter pw = new PrintWriter(osw);
			for(int i = 0;i<num;i++){
				pw.print(","+str[i]);
			}
			pw.close();
		}catch(IOException e){
			System.out.println(e);
		}
	}
	private static void addwrite(String filename,int num){
		File outputFile = new File(filename);
		FileOutputStream fos;
		try{
			fos = new FileOutputStream(outputFile,true);
			OutputStreamWriter osw = new OutputStreamWriter(fos);
			PrintWriter pw = new PrintWriter(osw);

			pw.print(","+num);
			pw.close();
		}catch(IOException e){
			System.out.println(e);
		}
	}

	private static void addwriteEND(String filename){
		File outputFile = new File(filename);
		FileOutputStream fos;
		try{
			fos = new FileOutputStream(outputFile,true);
			OutputStreamWriter osw = new OutputStreamWriter(fos);
			PrintWriter pw = new PrintWriter(osw);

			pw.println();
			pw.close();
		}catch(IOException e){
			System.out.println(e);
		}
	}

	public static class xpathCompiletoken {    //tokenの場所でコンパイル

		XPathExpression compiles(String id) throws XPathExpressionException{
			XPathExpression xpath =
					XPathFactory.newInstance()
					.newXPath()
					.compile("/corpus/document/sentence[@id='"+id+"']/sentenceanalyses/tokenizations/tokenization[@tokenizer='Charniak-Lease']/token");
			return(xpath);
		}
	}

	public static class xpathCompilePair {     //Pairの場所でコンパイル

		XPathExpression compiles(String id) throws XPathExpressionException{
			XPathExpression xpath =
					XPathFactory.newInstance()
					.newXPath()
					.compile("/corpus/document/sentence[@id='"+id+"']/pair");
			return(xpath);
		}
	}

	public static class xpathCompileDependency {     //dependencyの場所でコンパイル

		XPathExpression compiles(String id) throws XPathExpressionException{
			XPathExpression xpath =
					XPathFactory.newInstance()
					.newXPath()
					.compile("/corpus/document/sentence[@id='"+id+"']/sentenceanalyses/parses/parse[@parser='Charniak-Lease']/dependency");
			return(xpath);
		}
	}

	public static String[] RelWordSearch(int num,int num2,HashMap<Integer,token> map,HashSet<String> set){    //relation-wordがあるかどうか　あればそれの情報を登録
		String[] word = new String[4];
		String[] s = new String[2]; int i;
		word[0] = null; //word[3] = "low"; //relation-word　と　そのorigID と　その　charOffset と　flag(high,low)
		word[3] = "0"; //number of relation-words
		int count = 0; //count number of relation-words
		int small,big,SMALL,BIG;

		if(num > num2) {small = num2; big = num;}
		else {small = num; big = num2;}

		for(SMALL = small;SMALL < big;SMALL++){    //対象タンパク質間にrelation-wordがあるかどうか、あればwordに情報を登録（ただし１つだけ）、複数あればflagをhighにする。
			if(map.get(SMALL) != null){
				if((set.contains(map.get(SMALL).text)) == true ){
					count++;
					if(word[0] == null){
						s = (map.get(SMALL).id).split("\\_");
						i = (Integer.parseInt(s[1])) - 1;
						word[0] = map.get(SMALL).text;
						word[1] = String.valueOf(i);
						word[2] = String.valueOf(SMALL);
					}
				/*	else
						word[3] = "high";
				*/
				}
			}
		}

		for(SMALL = small;SMALL >= 0;SMALL--){    //ペアとなるタンパク質の前にrelation-wordがあるかどうか、あればwordに登録（ただし１つだけ）、複数あればflagをhighにする。
			if(map.get(SMALL) != null){
				if((set.contains(map.get(SMALL).text)) == true ){
					count++;
					if(word[0] == null){
						s = (map.get(SMALL).id).split("\\_");
						i = (Integer.parseInt(s[1])) - 1;
						word[0] = map.get(SMALL).text;
						word[1] = String.valueOf(i);
						word[2] = String.valueOf(SMALL);
					}
				/*	else
						word[3] = "high";
				*/
				}
			}
		}

		for(BIG = big;BIG < 400;BIG++){      //ペアとなるタンパク質の後にrelation-wordがあるかどうか、あればwordに登録（ただし１つだけ）、複数あればflagをhighにする。
			if(map.get(BIG) != null){
				if((set.contains(map.get(BIG).text)) == true ){
					count++;
					if(word[0] == null){
						s = (map.get(BIG).id).split("\\_");
						i = (Integer.parseInt(s[1])) - 1;
						word[0] = map.get(BIG).text;
						word[1] = String.valueOf(i);
						word[2] = String.valueOf(BIG);
					}
				/*	else
						word[3] = "high";
				*/
				}
			}
		}
		if(word[0] == null){
			word[0] = "NORELWORD";
	//		word[0] = "?";
			word[1] = "1";
			word[2] = "0";
		}

		word[3] = String.valueOf(count);
		return (word);
	}

	public static String[] Distance(int p1,int p2,String KeywordNum,String[] D,HashMap<Integer,token> map){ //feature D1,D2 order計算
		int word,a,b,c,P1,P2,f,m,s;
		String A,B,C;
		word = Integer.parseInt(KeywordNum);
		P1 = p1;
		P2 = p2;

		while(map.get(P1) == null){
			P1--;
		}
		while(map.get(P2) == null){
			P2--;
		}
		while(map.get(word) == null){
			word--;
		}
		A = map.get(P1).id;
		B = map.get(P2).id;
		C = map.get(word).id;

		String[] sID = A.split("\\_");
		a = (Integer.parseInt(sID[1]));

		String[] sid = B.split("\\_");
		b = (Integer.parseInt(sid[1]));

		String[] sId = C.split("\\_");
		c = (Integer.parseInt(sId[1]));

		D[3] = sID[1];
		D[4] = sid[1];
		D[5] = sId[1];

		/*D1 = D[0] = D_KP1
		  D2 = D[1] = D_KP2
		  D3 = D[2] = D_P1P2
		*
		*/
		if((a > b) && (a > c)){
			s = a;
			if(b > c){
			//	m = b; f = c;
				D[6] = "vab";
			}else{
			//	f = b; m = c;
				D[6] = "avb";
			}
		}else if(b > c){
			s = b;
			if(a > c){
			//	m = a; f = c;
				D[6] = "vab";
			}else{
			//	f = a; m = c;
				D[6] = "avb";
			}
		}else{
			s = c;
			D[6] = "abv";
		//	if(a > b){
			//	m = a; f = b;
		//	}else{
			//	f = a; m = b;
		//	}
		}

	/*	D[0] = String.valueOf(m - f);
		D[1] = String.valueOf(s - m);
		D[2] = String.valueOf(s - f);
	*/

		D[0] = String.valueOf(Math.abs(a - c));
		D[1] = String.valueOf(Math.abs(b - c));
		D[2] = String.valueOf(Math.abs(a - b));

		return (D);
	}

	public static String COMMA(int p1,int p2,String REL,HashMap<Integer,token> map){   //feature comma計算
		String comma = "null";
		String first = "n";
		String second = "n";
		int small=0,big=0,middle=0,rel=0;
		rel = Integer.parseInt(REL);

		if(p1>p2 && p1>rel && p2>rel)      {small=rel; middle=p2;  big=p1;}
		else if(p1>p2 && p1>rel && rel>p2) {small=p2;  middle=rel; big=p1;}
		else if(p2>p1 && p2>rel && p1>rel) {small=rel; middle=p1;  big=p2;}
		else if(p2>p1 && p2>rel && rel>p1) {small=p1;  middle=rel; big=p2;}
		else if(p1>p2)                     {small=p2;  middle=p1;  big=rel;}
		else if(p2>p1)                     {small=p1;  middle=p2;  big=rel;}

		for(;small < middle;small++){
			if(map.get(small) != null){
				if(map.get(small).text.equals(",")){
					first = "y";
				}
			}
		}
		for(;middle<big;middle++){
			if(map.get(middle) != null){
				if(map.get(middle).text.equals(",")){
					second = "y";
				}
			}
		}

		if(first.equals("n") && second.equals("n")) comma = "nn";
		else if(first.equals("n") && second.equals("y")) comma = "ny";
		else if(first.equals("y") && second.equals("n")) comma = "yn";
		else comma = "yy";

		return(comma);
	}

	public static String NOT(int p1,int p2,String REL,HashMap<Integer,token> map,HashSet<String> set){    //feature not計算
		String not = "false";
		int small=0,big=0,rel=0;
		rel = Integer.parseInt(REL);

		if(p1>p2 && p1>rel && p2>rel)      {small=rel; big=p1;}
		else if(p1>p2 && p1>rel && rel>p2) {small=p2;  big=p1;}
		else if(p2>p1 && p2>rel && p1>rel) {small=rel; big=p2;}
		else if(p2>p1 && p2>rel && rel>p1) {small=p1;  big=p2;}
		else if(p1>p2)                     {small=p2;  big=rel;}
		else if(p2>p1)                     {small=p1;  big=rel;}

		for(;small < big; small++){
			if(map.get(small) != null){
				if(set.contains(map.get(small).text) == true)
					not = "true";
			}
		}

		return(not);
	}

	public static String Breaker(int p1,int p2,String REL,HashMap<Integer,token> map,HashSet<String> set){  //feature breaker計算
		String breaker = "false";
		int small=0,big=0,rel=0;
		rel = Integer.parseInt(REL);

		if(p1>p2 && p1>rel && p2>rel)      {small=rel; big=p1;}
		else if(p1>p2 && p1>rel && rel>p2) {small=p2;  big=p1;}
		else if(p2>p1 && p2>rel && p1>rel) {small=rel; big=p2;}
		else if(p2>p1 && p2>rel && rel>p1) {small=p1;  big=p2;}
		else if(p1>p2)                     {small=p2;  big=rel;}
		else if(p2>p1)                     {small=p1;  big=rel;}

		for(;small < big; small++){
			if(map.get(small) != null){
				if(set.contains(map.get(small).text) == true)
					breaker = "true";
			}
		}
		return(breaker);
	}

	public static String Conditional(int p1,int p2,String REL,HashMap<Integer,token> map){                //feature conditional計算
		String conditional = "false";
		int small=0,rel=0,id=0,sid=0,SMALL=0;
		rel = Integer.parseInt(REL);

		if(p1<p2 && p1<rel)      small=p1;
		else if(p2<p1 && p2<rel) small=p2;
		else                     small=rel;

		SMALL = small;

		if(map.get(SMALL) != null){
			String[] sID = (map.get(SMALL).id).split("\\_");
			sid = (Integer.parseInt(sID[1]));
		}
		else {
			while(map.get(small) == null){
				small--;
			}
			String[] sID = (map.get(small).id).split("\\_");
			sid = (Integer.parseInt(sID[1]));
		}

		for(small = SMALL;small > 0;small--){
			if(map.get(small) != null){
				if(map.get(small).text.equals("IF") ||map.get(small).text.equals("WHETHER")){
					String[] ID = (map.get(small).id).split("\\_");
					id = (Integer.parseInt(ID[1]));
					if((sid - id) <= 3){
						conditional = "true";
					}
				}
			}
		}
		return(conditional);
	}

	public static String Prep(String REL,HashMap<Integer,token> map,HashSet<String> set){                 //feature prep計算
		String prep = "NA";
		int rel,sid,Rel,id;
		rel = Integer.parseInt(REL);
		Rel = rel;

		if(map.get(rel) != null){
			String[] sID = (map.get(rel).id).split("\\_");
			sid = (Integer.parseInt(sID[1]));
		}else{
			while(map.get(rel) == null){
				rel--;
			}
			String[] sID = (map.get(rel).id).split("\\_");
			sid = (Integer.parseInt(sID[1]));
		}
		for(rel = Rel;rel < 400;rel++){
			if(map.get(rel) != null){
				if(set.contains(map.get(rel).text) == true){
					String[] ID = (map.get(rel).id).split("\\_");
					id = (Integer.parseInt(ID[1]));
					if((sid - id) <= 3){
						prep = (map.get(rel).text).toLowerCase();
						break;
					}
				}
			}
		}
		return(prep);
	}

	public static String Which(int p1,int p2,String REL,HashMap<Integer,token> map){           //feature which計算
		String which = "false";
		int small=0,big=0,rel=0;
		rel = Integer.parseInt(REL);

		if(p1>p2 && p1>rel && p2>rel)      {small=rel; big=p1;}
		else if(p1>p2 && p1>rel && rel>p2) {small=p2;  big=p1;}
		else if(p2>p1 && p2>rel && p1>rel) {small=rel; big=p2;}
		else if(p2>p1 && p2>rel && rel>p1) {small=p1;  big=p2;}
		else if(p1>p2)                     {small=p2;  big=rel;}
		else if(p2>p1)                     {small=p1;  big=rel;}

		for(;small < big; small++){
			if(map.get(small) != null){
				if(map.get(small).text.equals("WHICH"))
					which = "true";
			}
		}
		return(which);
	}

	public static String But(int p1,int p2,String REL,HashMap<Integer,token> map){             //feature but計算
		String but = "false";
		int small=0,big=0,rel=0;
		rel = Integer.parseInt(REL);

		if(p1>p2 && p1>rel && p2>rel)      {small=rel; big=p1;}
		else if(p1>p2 && p1>rel && rel>p2) {small=p2;  big=p1;}
		else if(p2>p1 && p2>rel && p1>rel) {small=rel; big=p2;}
		else if(p2>p1 && p2>rel && rel>p1) {small=p1;  big=p2;}
		else if(p1>p2)                     {small=p2;  big=rel;}
		else if(p2>p1)                     {small=p1;  big=rel;}

		for(;small < big; small++){
			if(map.get(small) != null){
				if(map.get(small).text.equals("BUT"))
					but = "true";
			}
		}
		return(but);
	}

	private static String[] OverLap(String sentenceID, Entity e1, Entity e2,
			HashMap<String, Entity> entityMap) {                     //Overlapを計算する（[0]=Positiveに関与するものをtrue,[1]=Negativeに関与するものをtrue）
		String overlap[] = {"false","false","false","false","false","false"};
		int offset[][] = new int[30][2];
		Entity e;

		for (int j = 0; j < 30; j++) {
			for (int k = 0; k < 2; k++) {
				offset[j][k] = 0; //bug
			}
		}


		int i = 0;
		while(entityMap.containsKey(sentenceID+".e"+i)){
			e = entityMap.get(sentenceID+".e"+i);
			offset[i][0] = e.charoffsetF;            //Sentence内のcharoffsetを前後共に取得
			offset[i][1] = e.charoffsetR;
			i++;
		}

		int front = e1.charoffsetF;
		int rear = e1.charoffsetR;
		for(int j = 0; j<i;j++){
			if(front == offset[j][0]){ //bug
				if(rear != offset[j][1])
					if(rear > offset[j][1]){
						overlap[0] = "true";
					}
					else
						overlap[1] = "true";
			}
		}

		for(int j = 0; j<i;j++){ //追加
			if(rear == offset[j][1]){ //bug
				if(front != offset[j][0])
					if(front < offset[j][0]){
						overlap[2] = "true";
					}
					else
						overlap[3] = "true";
			}
		}


		front = e2.charoffsetF;
		rear = e2.charoffsetR;
		for(int j = 0; j<i;j++){
			if(front == offset[j][0]){
				if(rear != offset[j][1])
					if(rear > offset[j][1])
						overlap[0] = "true";
					else
						overlap[1] = "true";
			}
		}

		for(int j = 0; j<i;j++){ //追加
			if(rear == offset[j][1]){ //bug
				if(front != offset[j][0])
					if(front < offset[j][0]){
						overlap[2] = "true";
					}
					else
						overlap[3] = "true";
			}
		}

		if(e1.charoffsetF == e2.charoffsetF) overlap[4] = "true";//追加
		if(e1.charoffsetR == e2.charoffsetR) overlap[5] = "true";//追加

		return overlap;
	}


	public static Tree searchPRO(int e_char,String e_text,HashMap<Integer,token> map,List<Tree> list){    //sentenceからペアに使用するタンパク質を特定
		int num = list.size();
		int P1 = e_char;
		int tokenID;
		int pronum = 0;
		int dis = 100;
		int max = -1;
		int n = 0;
		String pro = e_text.toUpperCase();
		String words[] = null;
		String words_h[] = null;
		words = pro.split("[\\s\\.\\(\\)\\*\\~]");


		for(int i = 0;i<words.length;i++){                //一番大きなsplit後の単語を探索
			if(max < words[i].length()){
				max = words[i].length();
				n = i;
			}
		}


		while(map.get(P1) == null){
			P1--;
		}

		String A = map.get(P1).id;
		String[] sID = A.split("\\_");
		tokenID = (Integer.parseInt(sID[1]));

		for(int i = 0;i<num;i++){
			if(list.get(i).toString().contains("-")){
				words_h = list.get(i).toString().split("\\-");
				int NUM = words_h.length;
				for(int j = 0;j < NUM ;j++){
					if(words[n].equals(words_h[j])){
						if(dis > Math.abs(tokenID - i)){
							pronum = i;
							dis = Math.abs(tokenID - i);
						}
					}
				}
			}
			else if(words[n].equals(list.get(i).toString())){
				if(dis > Math.abs(tokenID - i)){
					pronum = i;
					dis = Math.abs(tokenID - i);
				}
			}
		}
		return(list.get(pronum));
	}

	private static int POSnumber_sum(Tree node, Tree root, HashSet<String> set) {         //POS Feature 計算（数値で）
		String words[] = {"",""};
		String POS[] = {"","","","","",""};
		int sum =0;
		//int n[] = {1,10,100,1000,10000,100000};
		int n[] = {1,5,25,125,625,3125};


		for(int i = 0;i < 6; i++){
			if(i == 0){
				node = node.ancestor(2, root);
			}
			else
				node = node.ancestor(1, root);
			words = node.nodeString().split("\\s");
			if(words[0].equals("S") || words[0].equals("ROOT") || words[0].equals("X")){
				POS[i] = words[0];
				break;
			}else{
				POS[i] = words[0];
			}
		}

		for(int j = 0; j < 6;j++){
			if(POS[j].length() == 0)
				sum  = sum + (0*n[j]);
			else if(set.contains(POS[j])){
				if(POS[j].equals("NP")) sum = sum + (1*n[j]);
				else if(POS[j].equals("PP")) sum = sum + (2*n[j]);
				else if(POS[j].equals("VP")) sum = sum + (3*n[j]);
				//else if(POS[j].equals("PRN") || POS[j].equals("PRT")) sum = sum + (4*n[j]);
				//else if(POS[j].equals("X")) sum = sum + (5*n[j]);
				//else if(POS[j].equals("ROOT")) sum = sum + (6*n[j]);
				//else if(POS[j].equals("S")) sum = sum + (7*n[j]);
				//else if(POS[j].equals("ADVP")) sum = sum + (8*n[j]);
				//else sum = sum + (9*n[j]);
			}else
				sum = sum + (4/*9*/*n[j]);
		}

		return sum;
	}

	public static void outputItiji(String tt){
		File outputFile = new File("itiji.txt");
		FileOutputStream fos;
		try{
			fos = new FileOutputStream(outputFile,true);
			OutputStreamWriter osw = new OutputStreamWriter(fos);
			PrintWriter pw = new PrintWriter(osw);

			pw.println(tt);

			pw.close();
		}catch(IOException e){
			System.out.println(e);
		}
	}


	public static String pattern_between(int p1, int p2, HashMap<Integer,token> tokenmap){  //between A and B　文型であるか否か
		String between = "BETWEEN";
		String result = "false";
		//small = p1; big = p2;
		boolean[] pattern = {false,false,false,false};

		for(int i = 0;(i <= p1 && pattern[0] == false) || i < p2+1;i++){
			if(tokenmap.get(i) != null){
				if(pattern[2] == true)
					if(i == p2)
						pattern[3] = true;
					else
						pattern[2] = false;
				if(pattern[1] == true)
					if(tokenmap.get(i).text.equals("AND"))
						pattern[2] = true;
					else
						pattern[1] = false;
				if(pattern[0] == true)
					if(i == p1)
						pattern[1] = true;
					else
						pattern[0] = false;
				if(tokenmap.get(i).text.equals(between))
					pattern[0] = true;
			}
		}

		if(pattern[3] == true)
			result = "true";

		return result;
	}

	public static String pattern_depend(int p1, int p2, HashMap<Integer,token> tokenmap){   //A depend B 文型であるか否か
		String[] depend = {"DEPEND","DEPENDS","DEPENDENT"};
		String result = "false";
		boolean[] pattern = {false,false,false};
		int wildcard = 0;

		for(int i = 0;i < p2+1; i++){
			if(tokenmap.get(i) != null){
				if(pattern[1] == true)
					if(i == p2)
						pattern[2] = true;
					else if(wildcard < 4) wildcard++;
					else pattern[1] = false;

				if(pattern[0] == true)
					if(tokenmap.get(i).text.equals(depend[0]) || tokenmap.get(i).text.equals(depend[1]) || tokenmap.get(i).text.equals(depend[2]))
						pattern[1] = true;
					else if(tokenmap.get(i).text.equals("-")) ;
					else pattern[0] = false;

				if(p1 == i)
					pattern[0] = true;
			}
		}

		if(pattern[2] == true)
			result = "true";

		return result;
	}

	public static String pattern_contiguous_protein(int p1,int p2, HashMap<Integer,token> tokenmap){     //蛋白質が隣接するパターン
		String[] para = {"-","/"};
		String result = "false";
		boolean[] pattern = {false,false};

		for(int i = 0;i<p2+1;i++){
			if(tokenmap.get(i) != null){
				if(pattern[0] == true)
					if(i == p2)
						pattern[1] = true;
					else if(tokenmap.get(i).text.equals(para[0]) || tokenmap.get(i).text.equals(para[1])) ;
					else pattern[0] = false;

				if(p1 == i)
					pattern[0] = true;
			}
		}

		if(pattern[1] == true)
			result = "true";

		return result;
	}

	public static String pattern_pararel_protein(int p1,int p2, HashMap<Integer,token> tokenmap){     //蛋白質が並列のようなパターン
		String[] para = {"AND","OR","("};
		String result = "false";
		boolean[] pattern = {false,false};

		if(p1==p2) return "true";//追加
	    /*<sentence id="AIMed.d45.s390" seqId="s390" text="A mutant form of human interferon-gamma (IFN-gamma SC1) that binds one IFN-gamma receptor alpha chain (IFN-gamma R alpha) has been designed and characterized.">

         <sentence id="AIMed.d45.s391" seqId="s391" text="IFN-gamma SC1 was derived by linking the two peptide chains of the IFN-gamma dimer by a seven-residue linker and changing His111 in the first chain to an aspartic acid residue.">
		*
		*/

		for(int i = 0;i<p2+1;i++){
			if(tokenmap.get(i) != null){
				if(pattern[0] == true && (p2 != (p1+1)) )
					if(i == p2)
						pattern[1] = true;
					else if(tokenmap.get(i).text.equals(para[0]) || tokenmap.get(i).text.equals(para[1]) || tokenmap.get(i).text.equals(para[2])) ;
					else pattern[0] = false;

				if(p1 == i)
					pattern[0] = true;
			}
		}

		if(pattern[1] == true)
			result = "true";

		return result;
	}
	private static String[] pattern11(int p1, int p2,
			HashMap<Integer, token> tokenMap, HashSet<String> nounSet,
			HashSet<String> verbSet,HashSet<String> pass_wordSet) {
		/* ------pattern_feature------
		 * 1:P*V*P
		 * 2:P*V*by*P
		 * 3:V of*P+by*P
		 * 4:V of*P*to*P
		 * 5:N of*P*(by|through)*P
		 * 6:N of*P*(with|to|on)*P
		 * 7:N between*P*and*P
		 * 8:complex between*P*and*P
		 * 9:complex of*P*and*P
		 * 10:P*form*complex with*P
		 * 11:P*P*N
		 */
		String pattern[] = {"false","false","false","false","false","false","false","false","false","false","false","false"};
		String by = "BY";String to = "TO";String of = "OF";String with = "WITH";String on = "ON";String through = "THROUGH";
		String between = "BETWEEN";String and = "AND";String complex = "COMPLEX";String complexes = "COMPLEXES";String complexed = "COMPLEXED";
		String form = "FORM";String forms = "FORMS";String formed = "FORMED";
		String depend = "DEPEND"; String depends = "DEPENDS"; String dependent = "DEPENDENT";
		int limit = 5;                                                    //wildcardの許容語数
		int limit_depend = 4;

		int verb_wildcard = 0;
		int noun_wildcard = 0;
		int p_wildcard = 0;
		int nb_wildcard = 0;
		int com_wildcard = 0;
		int form_wildcard = 0;
		int ppn_wildcard = 0;
		int depend_wildcard = 0;
		String v_pattern = null;               //verbPattern判定用
		String n_pattern = null;               //nounPattern判定用
		String p_pattern = null;               //pPattern判定用
		String com_pattern = null;               //comPattern判定用
		boolean[] Pflag = {false,false,false,false};
		boolean[] nounflag = {false,false,false,false,false};
		boolean[] verbflag = {false,false,false,false,false};
		boolean[] nbflag = {false,false,false,false,false};
		boolean[] comflag = {false,false,false,false,false};
		boolean[] formflag = {false,false,false,false,false};
		boolean[] ppnflag = {false,false,false};
		boolean[] dependflag = {false,false,false};


		for(int i = 0;i<p2+50;i++){
			if(tokenMap.get(i) != null){
				//---------------PvP_pattern1,2---------------------//
				if(Pflag[2] == true)
					if(i == p2){ Pflag[3] = true; p_wildcard=0;p_pattern = "2";}
					else if(pass_wordSet.contains(tokenMap.get(i).text));
					else if(p_wildcard < limit) p_wildcard++;
					else {Pflag[2] = false; p_wildcard=0;}

				if(Pflag[1] == true)
					if(i==p2){ Pflag[3] = true; p_wildcard=0; p_pattern = "1";}
					else if(tokenMap.get(i).text.equals(by)){ Pflag[2] = true; p_wildcard=0;}
					else if(pass_wordSet.contains(tokenMap.get(i).text));
					else if(p_wildcard < limit) p_wildcard++;
					else {Pflag[1] = false; p_wildcard=0;}

				if(Pflag[0] == true)
					if(pass_wordSet.contains(tokenMap.get(i).text));
					else if(verbSet.contains(tokenMap.get(i).text)) {Pflag[1] = true;p_wildcard = 0;}
					else if(p_wildcard < limit) p_wildcard++;
					else {Pflag[0] = false;p_wildcard=0;}

				if(i == p1)
					Pflag[0] = true;
				//-------------------ここまで------------------------//
				//---------------verb_pattern3,4---------------------//
				if(verbflag[3] == true)
					if(i == p2){ verbflag[4] = true; verb_wildcard=0;}
					else if(pass_wordSet.contains(tokenMap.get(i).text));
					else if(verb_wildcard < limit) verb_wildcard++;
					else {verbflag[3] = false; verb_wildcard=0;}

				if(verbflag[2] == true)
					if(tokenMap.get(i).text.equals(by)){ verbflag[3] = true; verb_wildcard=0; v_pattern = by;}
					else if(tokenMap.get(i).text.equals(to)){ verbflag[3] = true; verb_wildcard=0; v_pattern = to;}
					else if(pass_wordSet.contains(tokenMap.get(i).text));
					else if(verb_wildcard < limit) verb_wildcard++;
					else {verbflag[2] = false; verb_wildcard=0;}

				if(verbflag[1] == true)
					if(i == p1){ verbflag[2] = true; verb_wildcard=0;}
					else if(pass_wordSet.contains(tokenMap.get(i).text));
					else if(verb_wildcard < limit) verb_wildcard++;
					else {verbflag[1] = false;verb_wildcard=0;}

				if(verbflag[0] == true)
					if(pass_wordSet.contains(tokenMap.get(i).text));
					else if(tokenMap.get(i).text.equals(of)) verbflag[1] = true;
					else verbflag[0] = false;

				if(verbSet.contains(tokenMap.get(i).text))
					verbflag[0] = true;
				//-------------------ここまで------------------------//
				//---------------noun_pattern5,6---------------------//
				if(nounflag[3] == true)
					if(i == p2){ nounflag[4] = true; noun_wildcard=0;}
					else if(pass_wordSet.contains(tokenMap.get(i).text));
					else if(noun_wildcard < limit) noun_wildcard++;
					else {nounflag[3] = false; noun_wildcard=0;}

				if(nounflag[2] == true)
					if(tokenMap.get(i).text.equals(by)||tokenMap.get(i).text.equals(through)){ nounflag[3] = true; noun_wildcard=0; n_pattern = by;}
					else if(tokenMap.get(i).text.equals(to)||tokenMap.get(i).text.equals(with)||tokenMap.get(i).text.equals(on))
					{ nounflag[3] = true; noun_wildcard=0; n_pattern = to;}
					else if(pass_wordSet.contains(tokenMap.get(i).text));
					else if(noun_wildcard < limit) noun_wildcard++;
					else {nounflag[2] = false; noun_wildcard=0;}

				if(nounflag[1] == true)
					if(i == p1){ nounflag[2] = true; noun_wildcard=0;}
					else if(pass_wordSet.contains(tokenMap.get(i).text));
					else if(noun_wildcard < limit) noun_wildcard++;
					else {nounflag[1] = false;noun_wildcard=0;}

				if(nounflag[0] == true)
					if(pass_wordSet.contains(tokenMap.get(i).text));
					else if(tokenMap.get(i).text.equals(of)) nounflag[1] = true;
					else nounflag[0] = false;

				if(nounSet.contains(tokenMap.get(i).text))
					nounflag[0] = true;
				//-------------------ここまで--------------------------//
				//---------------pattern7---------------------------//
				if(nbflag[3] == true)
					if(i == p2){ nbflag[4] = true; nb_wildcard=0;}
					else if(pass_wordSet.contains(tokenMap.get(i).text));
					else if(nb_wildcard < limit) nb_wildcard++;
					else {nbflag[3] = false; nb_wildcard=0;}

				if(nbflag[2] == true)
					if(tokenMap.get(i).text.equals(and)){ nbflag[3] = true; nb_wildcard=0;}
					else if(pass_wordSet.contains(tokenMap.get(i).text));
					else if(nb_wildcard < limit) nb_wildcard++;
					else {nbflag[2] = false; nb_wildcard=0;}

				if(nbflag[1] == true)
					if(i == p1){ nbflag[2] = true; nb_wildcard=0;}
					else if(pass_wordSet.contains(tokenMap.get(i).text));
					else if(nb_wildcard < limit) nb_wildcard++;
					else {nbflag[1] = false;nb_wildcard=0;}

				if(nbflag[0] == true)
					if(pass_wordSet.contains(tokenMap.get(i).text));
					else if(tokenMap.get(i).text.equals(between)) nbflag[1] = true;
					else nbflag[0] = false;

				if(nounSet.contains(tokenMap.get(i).text))
					nbflag[0] = true;
				//-------------------ここまで--------------------------//
				//---------------pattern8,9---------------------//
				if(comflag[3] == true)
					if(i == p2){ comflag[4] = true; com_wildcard=0;}
					else if(pass_wordSet.contains(tokenMap.get(i).text));
					else if(com_wildcard < limit) com_wildcard++;
					else {comflag[3] = false; com_wildcard=0;}

				if(comflag[2] == true)
					if(tokenMap.get(i).text.equals(and)){ comflag[3] = true; com_wildcard=0; }
					else if(pass_wordSet.contains(tokenMap.get(i).text));
					else if(com_wildcard < limit) com_wildcard++;
					else {comflag[2] = false; com_wildcard=0;}

				if(comflag[1] == true)
					if(i == p1){ comflag[2] = true; com_wildcard=0;}
					else if(pass_wordSet.contains(tokenMap.get(i).text));
					else if(com_wildcard < limit) com_wildcard++;
					else {comflag[1] = false;com_wildcard=0;}

				if(comflag[0] == true)
					if(pass_wordSet.contains(tokenMap.get(i).text));
					else if(tokenMap.get(i).text.equals(between)){ comflag[1] = true;com_pattern = between;}
					else if(tokenMap.get(i).text.equals(of)){comflag[1] = true;com_pattern = of;}
					else comflag[0] = false;

				if(tokenMap.get(i).text.equals(complex)||tokenMap.get(i).text.equals(complexes)||tokenMap.get(i).text.equals(complexed))
					comflag[0] = true;
				//-------------------ここまで--------------------------//
				//---------------pattern10---------------------//
				if(formflag[3] == true)
					if(i == p2){ formflag[4] = true; form_wildcard=0;}
					else if(pass_wordSet.contains(tokenMap.get(i).text));
					else if(form_wildcard < limit) form_wildcard++;
					else {formflag[3] = false; form_wildcard=0;}

				if(formflag[2] == true)
					if(tokenMap.get(i).text.equals(with)) formflag[3] = true;
					else if(pass_wordSet.contains(tokenMap.get(i).text));
					else formflag[2] = false;

				if(formflag[1] == true)
					if(tokenMap.get(i).text.equals(complex)||tokenMap.get(i).text.equals(complexes)||tokenMap.get(i).text.equals(complexed))
					{ formflag[2] = true; form_wildcard=0;}
					else if(pass_wordSet.contains(tokenMap.get(i).text));
					else if(form_wildcard < limit) form_wildcard++;
					else {formflag[1] = false; form_wildcard=0;}

				if(formflag[0] == true)
					if(pass_wordSet.contains(tokenMap.get(i).text));
					else if(tokenMap.get(i).text.equals(form)||tokenMap.get(i).text.equals(forms)||tokenMap.get(i).text.equals(formed))
					{formflag[1] = true;form_wildcard = 0;}
					else if(form_wildcard < limit) form_wildcard++;
					else {formflag[0] = false;form_wildcard=0;}

				if(i == p1)
					formflag[0] = true;
				//-------------------ここまで------------------------//
				//---------------ppn_pattern11---------------------//
				if(ppnflag[1] == true)
					if(nounSet.contains(tokenMap.get(i).text)){ ppnflag[2] = true; ppn_wildcard=0;}
					else if(pass_wordSet.contains(tokenMap.get(i).text));
					else if(ppn_wildcard < limit) ppn_wildcard++;
					else {ppnflag[1] = false; ppn_wildcard=0;}

				if(ppnflag[0] == true)
					if(pass_wordSet.contains(tokenMap.get(i).text));
					else if(i == p2) {ppnflag[1] = true;ppn_wildcard = 0;}
					else if(ppn_wildcard < limit) ppn_wildcard++;
					else {ppnflag[0] = false;ppn_wildcard=0;}

				if(i == p1)
					ppnflag[0] = true;
				//-------------------ここまで------------------------//
				//---------------ppn_pattern11---------------------//
				if(dependflag[1] == true)
					if(i == p2){ dependflag[2] = true; depend_wildcard=0;}
					else if(pass_wordSet.contains(tokenMap.get(i).text));
					else if(depend_wildcard < limit_depend) depend_wildcard++;
					else {dependflag[1] = false; depend_wildcard=0;}

				if(dependflag[0] == true)
					if(pass_wordSet.contains(tokenMap.get(i).text));
					else if(tokenMap.get(i).text.equals(depend)||tokenMap.get(i).text.equals(depends)||tokenMap.get(i).text.equals(dependent))
						dependflag[1] = true;
					else dependflag[0] = false;

				if(i == p1)
					dependflag[0] = true;
				//-------------------ここまで------------------------//
			}
		}

		if(Pflag[3] == true)
			if(p_pattern.equals("1")) pattern[0] = "true";    //pattern1
			else pattern[1] = "true";                         //pattern2
		if(verbflag[4] == true)
			if(v_pattern.equals(by)) pattern[2] = "true";     //pattern3
			else pattern[3] = "true";                         //pattern4
		if(nounflag[4] == true)
			if(n_pattern.equals(by)) pattern[4] = "true";     //pattern5
			else pattern[5] = "true";                         //pattern6
		if(nbflag[4] == true)
			pattern[6] = "true";                              //pattern7
		if(comflag[4] == true)
			if(com_pattern.equals(between)) pattern[7] = "true"; //pattern8
			else pattern[8] = "true";                            //pattern9
		if(formflag[4] == true)
			pattern[9] = "true";                              //pattern10
		if(ppnflag[2] == true)
			pattern[10] = "true";                             //pattern11
		if(dependflag[2] == true)
			pattern[11] = "true";                             //pattern12

		return pattern;
	}



	public static void main(String srgs[])throws Exception{
		HashMap<String,Entity> EntityMap = new HashMap<String,Entity>();
		HashMap<Integer,token> TokenMap = new HashMap<Integer,token>();
		HashSet<String> rel_hs = new HashSet<String>();
		HashSet<String> NotSet = new HashSet<String>();
		HashSet<String> BreakerSet = new HashSet<String>();
		HashSet<String> PrepSet = new HashSet<String>();
		HashSet<String> POSSet = new HashSet<String>();
		HashSet<String> NounSet = new HashSet<String>();
		HashSet<String> VerbSet = new HashSet<String>();
		HashSet<String> PasswordSet = new HashSet<String>();


		Properties props = new Properties();
		props.put("annotators","tokenize, ssplit, pos, lemma, ner, parse, dcoref");
		StanfordCoreNLP pipeline = new StanfordCoreNLP(props);

		//String filename = "Features.arff"; //出力ファイル
		String filename = "ARFFFiles20170613/" + corpus  + ".arff";
		//String filename = corpus  + "Test.arff";


		String nameID = "";
		if(corpus.equals("IEPA")){
			nameID = "origID";          //入力ファイルのentity内のIDの名前
		}else if(corpus.equals("AImed")) {
			nameID = "seqId";
		}else{
			nameID = "origId";
		}



		Document document = DocumentBuilderFactory.newInstance().
				newDocumentBuilder().parse(new File("XMLFiles/" + corpus +".xml"));  //入力ファイル


		rel_hs = WordsSet("RelWord.txt",rel_hs);                  //relation-word保存用
		NotSet = WordsSet("NotWord.txt",NotSet);                  //not-word保存用
		BreakerSet = WordsSet("BreakerWord.txt",BreakerSet);      //breaker-word保存用
		PrepSet = WordsSet("PrepWord.txt",PrepSet);               //prep-word保存用
		POSSet = WordsSet("POSWord.txt",POSSet);                  //POS-word保存用
		NounSet = WordsSet("Noun_keyword.txt",NounSet);           //noun-word保存用
		VerbSet = WordsSet("Verb_keyword.txt",VerbSet);           //verb-word保存用
		PasswordSet = WordsSet("pass_word.txt",PasswordSet);           //verb-word保存用


		String tt = "A";
		String[] SentenceID = new String[5000];           //sentenceID保存用
		String[] SentenceTEXT = new String[5000];         //sentenceTEXT保存用
		String[] keyword = new String[4];                 //relation-word　と　そのorigID と　その　charOffset と　flag(high,low)
		String[] D = new String[7];                       //Feature D1 D2 D3 Pro_position1,2, position of keyword, order用の配列
		String comma = "null";                            //Feature comma 用の変数
		String not = "null";                              //Feature not   用の変数
		String breaker = "null";                          //Feature breaker 用の変数
		String conditional = "null";                      //Feature conditional 用の変数
		String prep = "null";                             //Feature prep 用の変数
		String which = "null";                            //Feature which 用の変数
		String but = "null";                              //Feature but 用の変数
		int H1 = 0;                                       //Heigt　Feature
		int H2 = 0;
		int H3 = 0;
		int POS1 = 0;                                     //POS Feature
		int POS2 = 0;
		int POS3 = 0;
		String between = "null";
		String contiguous = "null";
		String para = "null";
		String Overlap[] = {null,null};                         //Feature Overlap　用 ([0] = Positiveに関与、[1] = Negativeに関与)
		String pattern[] = {null,null,null,null,null,null,null,null,null,null,null,null};
		String Bind =  null;
		String Interact = null;
		String Known = null;
		String Regulate = null;
		String Induce = null;
		String Stimulate = null;
		String Associate = null;


		LexicalizedParser lp = new LexicalizedParser("englishPCFG.ser.gz");    //パース用辞書設定


		//この２つはクラスにするほうがよいかも
		XPathExpression SentencePath = XPathFactory.newInstance().newXPath().compile("/corpus/document/sentence");
		XPathExpression EntityPath = XPathFactory.newInstance().newXPath().compile("//entity");



		NodeList NodeSentence = (NodeList) SentencePath.evaluate(document, XPathConstants.NODESET);
		NodeList NodeEntity = (NodeList) EntityPath.evaluate(document, XPathConstants.NODESET);

		for(int i = 0;i < NodeEntity.getLength(); ++i){     //Entityを作成し、ハッシュマップへ登録（keyはid)
			Node node = NodeEntity.item(i);
			NamedNodeMap attribute = node.getAttributes();
			String text = attribute.getNamedItem("text").getNodeValue();
			String origId = attribute.getNamedItem(nameID).getNodeValue();
			String id = attribute.getNamedItem("id").getNodeValue();
			String charoffset = attribute.getNamedItem("charOffset").getNodeValue();

			String[] Offset = charoffset.split("\\-");
			int offsetF = Integer.parseInt(Offset[0]);

		/*	String Offset1 = Offset[1];
			if (Offset[1].contains(",")){
				Offset1 = Offset[1].split("\\,")[1]; //0->1
				System.out.println("Offset1 = " + Offset1 + "; OffsetF = " + offsetF + "; text = " +text + "; id = " + id);
			}
			if (Offset[1].contains("-")){
				Offset1 = Offset[1].split("\\-")[1]; //
				System.out.println("Offset1 = " + Offset1 + "; OffsetF = " + offsetF + "; text = " +text + "; id = " + id);
			}
		*/


		//	int offsetR = Integer.parseInt(Offset1);
			int offsetR = Integer.parseInt(Offset[Offset.length-1]);

			System.out.println("Offset.length = " + Offset.length + "; offsetR = " + offsetR + "; offsetF = " + offsetF + "; text = " +text + "; id = " + id);

			Entity e = new Entity(origId,text,offsetF,offsetR);
			EntityMap.put(id, e);
		}

		for(int j = 0;j < NodeSentence.getLength(); ++j){      //sentence毎にidを配列に保存
			Node n = NodeSentence.item(j);
			NamedNodeMap Attribute = n.getAttributes();
			String ID = Attribute.getNamedItem("id").getNodeValue();
			String Text = Attribute.getNamedItem("text").getNodeValue();
			String TEXT = Text.toUpperCase();

			SentenceID[j] = ID;
			SentenceTEXT[j] = TEXT;
		}



		write(filename,"RelWord.txt","stemming_relword.txt");                   //arffファイルの冒頭の定義の部分を記述


		for(int i = 0;SentenceID[i] != null;i++){          //sentence毎に解析するループ　
			xpathCompiletoken cpTOKEN = new xpathCompiletoken();
			xpathCompilePair cpPAIR = new xpathCompilePair();
			xpathCompileDependency cpDEPENDENCY = new xpathCompileDependency();

			NodeList NodeToken = (NodeList) cpTOKEN.compiles(SentenceID[i]).evaluate(document,XPathConstants.NODESET);
			NodeList NodePair = (NodeList) cpPAIR.compiles(SentenceID[i]).evaluate(document,XPathConstants.NODESET);

			NodeList NodeDependency = (NodeList) cpDEPENDENCY.compiles(SentenceID[i]).evaluate(document,XPathConstants.NODESET);

			// 構文解析
			Tree parse = (Tree) lp.apply(SentenceTEXT[i]);
			// 表示フォーマットを設定
			//TreePrint tp = new TreePrint("penn");
			//tp.printTree(parse);
			List<Tree> list = parse.getLeaves();       //テキストの葉のリスト作成

			for(int j = 0;j < NodeToken.getLength(); ++j){   // tokenを作成し、ハッシュマップに登録（keyはcharOffsetの最初の数字)
				Node n2 = NodeToken.item(j);
				NamedNodeMap att = n2.getAttributes();
				String text = att.getNamedItem("text").getNodeValue();
				String TEXT = text.toUpperCase();
				String POS = att.getNamedItem("POS").getNodeValue();
				String id = att.getNamedItem("id").getNodeValue();
				String charOffset = att.getNamedItem("charOffset").getNodeValue();

				String[] Offset = charOffset.split("\\-");
				int offset = Integer.parseInt(Offset[0]);

				token t = new token(POS,TEXT,id);
				TokenMap.put(offset, t);
			}


			for(int k = 0;k < NodePair.getLength(); k++){      //pair毎に特徴と正解データを作成し、arffファイルに追加で書き込み.
				Node n3 = NodePair.item(k);
				NamedNodeMap at = n3.getAttributes();
				String pair_e1 = at.getNamedItem("e1").getNodeValue();
				String pair_e2 = at.getNamedItem("e2").getNodeValue();
				String interaction = at.getNamedItem("interaction").getNodeValue();

				Entity e1 = EntityMap.get(pair_e1);      //正解となるペアのEntityをハッシュマップから取得
				Entity e2 = EntityMap.get(pair_e2);


				//-----------------------------------------------------------------------------------------------------------------------------//
				keyword = RelWordSearch(e1.charoffsetF,e2.charoffsetF,TokenMap,rel_hs);
				Annotation Document = new Annotation(keyword[0].toLowerCase());                                //keywordのlemmatize
				pipeline.annotate(Document);
				List<CoreMap> sentences = Document.get(SentencesAnnotation.class);
				for(CoreMap sen: sentences){
					tt = "a";
					for(CoreLabel token: sen.get(TokensAnnotation.class)){
						String lemma = token.getString(LemmaAnnotation.class);
						if(tt.length() <= lemma.length())
							tt = lemma;
					}
				}
				outputItiji(tt);

				char[] w = new char[501];
				Stemmer s = new Stemmer();
				FileInputStream in = new FileInputStream("itiji.txt");

				try{
					while(true){
						int ch = in.read();


						if (Character.isLetter((char) ch)){
							int j = 0;
							while(true){
								ch = Character.toLowerCase((char) ch);
								w[j] = (char) ch;
								if (j < 500) j++;
								ch = in.read();
								if (!Character.isLetter((char) ch)){
									/* to test add(char ch) */
									for (int c = 0; c < j; c++) s.add(w[c]);

									/* or, to test add(char[] w, int j) */
									/* s.add(w, j); */

									s.stem();
									{  String u;

									/* and now, to test toString() : */
									u = s.toString();

									/* to test getResultBuffer(), getResultLength() : */
									/* u = new String(s.getResultBuffer(), 0, s.getResultLength()); */

									keyword[0] = u;
									}
									break;
								}
							}
						}
						if (ch < 0) break;
						//System.out.print((char)ch);
					}
				}
				catch (IOException e){
					System.out.println("error reading ");
				}
				//--------------------------------------------------------------------------------------//

				D = Distance(e1.charoffsetF,e2.charoffsetF,keyword[2],D,TokenMap);             //距離feature計算
				comma = COMMA(e1.charoffsetF,e2.charoffsetF,keyword[2],TokenMap);     //comma Feature計算
				not = NOT(e1.charoffsetF,e2.charoffsetF,keyword[2],TokenMap,NotSet);  //not Feature計算
				breaker = Breaker(e1.charoffsetF,e2.charoffsetF,keyword[2],TokenMap,BreakerSet);  //breaker feature計算
				conditional = Conditional(e1.charoffsetF,e2.charoffsetF,keyword[2],TokenMap);     //conditional feature　計算
				prep = Prep(keyword[2],TokenMap,PrepSet);                                       //prep feature 計算
				which = Which(e1.charoffsetF,e2.charoffsetF,keyword[2],TokenMap);  //which feature計算
				but = But(e1.charoffsetF,e2.charoffsetF,keyword[2],TokenMap);  //but feature計算
				Overlap = OverLap(SentenceID[i],e1,e2,EntityMap);


				Tree PRO1tree = searchPRO(e1.charoffsetF,e1.text,TokenMap,list);
				Tree PRO2tree = searchPRO(e2.charoffsetF,e2.text,TokenMap,list);
				Tree PRO3tree = searchPRO(Integer.parseInt(keyword[2]),keyword[0],TokenMap,list);
				H1 = parse.depth(PRO1tree);
				H2 = parse.depth(PRO2tree);
				H3 = parse.depth(PRO3tree);
				POS1 = POSnumber_sum(PRO1tree,parse,POSSet);
				POS2 = POSnumber_sum(PRO2tree,parse,POSSet);
				POS3 = POSnumber_sum(PRO3tree,parse,POSSet);
				between = pattern_between(e1.charoffsetF,e2.charoffsetF,TokenMap);

				contiguous = pattern_contiguous_protein(e1.charoffsetF,e2.charoffsetF,TokenMap);
				para = pattern_pararel_protein(e1.charoffsetF,e2.charoffsetF,TokenMap);

			/*	Bind = sec_key_bind(e1,e2,TokenMap,D,keyword[2]);
				Interact = sec_key_inter(e1,e2,TokenMap,D,keyword[2]);
				Known = sec_key_known(e1,e2,TokenMap,D,keyword[2]);
				Regulate = sec_key_regu(e1,e2,TokenMap,D,keyword[2]);
				Induce = sec_key_induce(e1,e2,TokenMap,D,keyword[2]);
				Stimulate = sec_key_stim(e1,e2,TokenMap,D,keyword[2]);
				Associate = sec_key_asso(e1,e2,TokenMap,D,keyword[2]);
			*/

				pattern = pattern11(e1.charoffsetF,e2.charoffsetF,TokenMap,NounSet,VerbSet,PasswordSet);

				//between = pattern_between(e1.charoffset,e2.charoffset,TokenMap);
				//depend = pattern_depend(e1.charoffset,e2.charoffset,TokenMap);
				//para = pattern_pararel_protein(e1.charoffset,e2.charoffset,TokenMap);

				//System.out.println(keyword[0]+","+SentenceID[i]);

//**********************************************************************************************************************
/*Dependency Parsing*/

				// Dependency解析
				List<TypedDependency> tdl = new ArrayList<TypedDependency>();
				List<String> ListGov = new ArrayList<String>();
				List<String> ListDep = new ArrayList<String>();

				for(int j = 0;j < NodeDependency .getLength(); ++j){   // Dependency を作成し、ハッシュマップに登録（keyはcharOffsetの最初の数字)
					Node nd = NodeDependency .item(j);
					NamedNodeMap attd = nd.getAttributes();
					String type = attd.getNamedItem("type").getNodeValue();

					String t1 = attd.getNamedItem("t1").getNodeValue();
					String T1 = t1.split("\\_")[1];

					String t2 = attd.getNamedItem("t2").getNodeValue();
					String T2 = t2.split("\\_")[1];


					Node x1 = NodeToken.item(Integer.parseInt(T1)-1);
					NamedNodeMap att1 = x1.getAttributes();
					String text1 = att1.getNamedItem("text").getNodeValue();

					Node x2 = NodeToken.item(Integer.parseInt(T2)-1);
					NamedNodeMap att2 = x2.getAttributes();
					String text2 = att2.getNamedItem("text").getNodeValue();


					String strGov = text1 + "-" + T1;
					String strDep = text2 + "-" + T2;
					TypedDependency TypeDep = TreeOperator.createDependency(type, strGov, strDep);
					tdl.add(TypeDep);

					ListGov.add(strGov);
					ListDep.add(strDep);
				}

				/*find ROOT*/
				String testROOT = "";
				for(int t=0; t<ListGov.size();t++){
					testROOT = ListGov.get(t);
					if(!ListDep.contains(testROOT))
						break;
				}

				System.out.println("testROOT= " +testROOT);
				TypedDependency TypeDepROOT = TreeOperator.createDependency("root","ROOT-0", testROOT);
				tdl.add(TypeDepROOT);

				//String[] tROOT = testROOT.split("\\-");
				//int indexROOT = Integer.parseInt(tROOT[1]);



			/*	 LexicalizedParser lparser = new LexicalizedParser("englishPCFG.ser.gz");    //パース用辞書設定
				  TreebankLanguagePack tlp = new PennTreebankLanguagePack();
				  // Uncomment the following line to obtain original Stanford Dependencies
				  // tlp.setGenerateOriginalDependencies(true);
				  GrammaticalStructureFactory gsf = tlp.grammaticalStructureFactory();
				  //String sent = "These results suggest that YfhP may act as a negative regulator for the transcription of yfhQ, yfhR, sspE and yfhP." ;
				  //Tree parse = lparser.apply(Sentence.toWordList(sent));
				  Tree parseDependency = lparser.apply(SentenceTEXT[i]);
				  GrammaticalStructure gs = gsf.newGrammaticalStructure(parseDependency);



				  //Collection<TypedDependency> tdl = gs.typedDependenciesCCprocessed();

				  List<TypedDependency> tdl = gs.typedDependenciesCCprocessed();
			*/

				  /*D1 = D[0] = D_KP1
				   *D2 = D[1] = D_KP2
				   *D3 = D[2] = D_P1P2
				   *P_postion1 = D[3] (position of protein P1)
				   *P_postion2 = D[4] (position of protein P2)
				   *P_postionK = D[5] (position of protein keyword K)
				   *order = D[6] = avb, abv, vab
				   */
				  //List<TypedDependency> str_dep = stringToDependencies(tdl);
				  //String str_rel = TreeOperator.dependencyPath(5,20,tdl);//shortest dependency path
				  System.out.println("Test0: " +SentenceTEXT[i]);
				  System.out.println("Test1: " + tdl);
				  //System.out.println("Test2: " +directPath(0,22,tdl));

				  int po1 = Integer.parseInt(D[3]);
				  int po2 = Integer.parseInt(D[4]);
				  int poK = Integer.parseInt(D[5]);

				  /*Number of steps of dependency paths, left */
				  int Dep_KP1=0, negative_KP1=0, positive_KP1=0;
				  int Dep_KP2=0, negative_KP2=0, positive_KP2=0;
				  int Dep_P1P2=0, negative_P1P2=0, positive_P1P2=0;
				  int Dep_P1Root=0, Dep_P2Root=0, Dep_KRoot=0;

				  Dep_KP1 = TreeOperator.CountStepsDependencyPath(Math.min(poK, po1),Math.max(poK, po1),tdl);
				  negative_KP1 = TreeOperator.CountNegativeStepsDependencyPath(Math.min(poK, po1),Math.max(poK, po1),tdl);
				  positive_KP1 = TreeOperator.CountPositiveStepsDependencyPath(Math.min(poK, po1),Math.max(poK, po1),tdl);

				  Dep_KP2 = TreeOperator.CountStepsDependencyPath(Math.min(poK, po2),Math.max(poK, po2),tdl);
				  negative_KP2 = TreeOperator.CountNegativeStepsDependencyPath(Math.min(poK, po2),Math.max(poK, po2),tdl);
				  positive_KP2 = TreeOperator.CountPositiveStepsDependencyPath(Math.min(poK, po2),Math.max(poK, po2),tdl);

				  Dep_P1P2 = TreeOperator.CountStepsDependencyPath(Math.min(po1, po2),Math.max(po1, po2),tdl);
				  negative_P1P2 = TreeOperator.CountNegativeStepsDependencyPath(Math.min(po1, po2),Math.max(po1, po2),tdl);
				  positive_P1P2 = TreeOperator.CountPositiveStepsDependencyPath(Math.min(po1, po2),Math.max(po1, po2),tdl);



				  Dep_P1Root = TreeOperator.CountStepsDependencyPath(0,po1,tdl);
				  Dep_P2Root = TreeOperator.CountStepsDependencyPath(0,po2,tdl);
				  Dep_KRoot = TreeOperator.CountStepsDependencyPath(0,poK,tdl);
//**********************************************************************************************************************

				addwriteFirst(filename, keyword[0]);
				addwrite(filename, D);
				addwrite(filename, keyword[3]);
			//	addwrite(filename, comma);
				addwrite(filename, not);
			//	addwrite(filename, breaker);
			//	addwrite(filename, conditional);
				addwrite(filename, prep);
			//	addwrite(filename, which);
			//	addwrite(filename, but);
				addwrite(filename, H1);
				addwrite(filename, H2);
				addwrite(filename, H3);

			/* addwrite(filename, POS1);
				addwrite(filename, POS2);
				addwrite(filename, POS3);
			*/
				addwrite(filename, Overlap[0]);
				addwrite(filename, Overlap[1]);
				addwrite(filename, Overlap[2]);
				addwrite(filename, Overlap[3]);
				addwrite(filename, Overlap[4]);
				addwrite(filename, Overlap[5]);
				addwrite(filename, pattern); //patterns matching
				addwrite(filename, between);
			    addwrite(filename, contiguous);
				addwrite(filename, para);

			/*	addwrite(filename, Bind);
				addwrite(filename, Interact);
				addwrite(filename, Known);
				addwrite(filename, Regulate);
				addwrite(filename, Induce);
				addwrite(filename, Stimulate);
				addwrite(filename, Associate);
			*/
				addwrite(filename, Dep_KP1);
				addwrite(filename, negative_KP1);
				addwrite(filename, positive_KP1);

				addwrite(filename, Dep_KP2);
				addwrite(filename, negative_KP2);
				addwrite(filename, positive_KP2);

				addwrite(filename, Dep_P1P2);
				addwrite(filename, negative_P1P2);
				addwrite(filename, positive_P1P2);

				addwrite(filename, Dep_P1Root);
				addwrite(filename, Dep_P2Root);
				addwrite(filename, Dep_KRoot);


				addwrite(filename, interaction);
				addwriteEND(filename);


				//addwrite("./Features.arff",keyword[0],D,keyword[3],comma,not,breaker,conditional,prep,which,but,H1,H2,H3,POS1,POS2,POS3,interaction);
			}
			//各sentence解析毎にtokenMapを初期化
			TokenMap.clear();
		}

		DataSource source = new DataSource(filename);
		Instances instances = source.getDataSet();
		instances.setClassIndex(instances.numAttributes() - 1);

		//Classifier classifier = new J48();        //J48の場合
		Classifier classifier = new SMO();  //SMOの場合
		//classifier.buildClassifier(instances);

		Random random = new Random();
		//Evaluation eval = new Evaluation(instances);
		Evaluation eval = new Evaluation(instances);
		//d = eval.evaluateModel(classifier, instances);
		//d = eval.evaluateModel(classifier, test);
		eval.crossValidateModel(classifier, instances, 10,/* new Random(1)*/random);
		System.out.println(eval.toSummaryString("\nResults\n======\n", false));
		System.out.println(eval.toClassDetailsString());

		double p = eval.precision(0);
		double r = eval.recall(0);
		double f = eval.fMeasure(0);

		System.out.println("F値:" + f +"  Recall:"+ r + "  Precdition:" +p);


	}


	private static String sec_key_induce(Entity e1, Entity e2,
			HashMap<Integer, token> TokenMap, String[] D, String keynum) {
			int bownum = 0;
			String re = "false";
			HashSet<String> set = new HashSet<String>();

			set.add("INDUCE");
			set.add("INDUCED");
			set.add("INDUCING");

			bownum = 0;

			if(D[5].equals("avb")){
				bownum = Integer.parseInt(keynum);
				while(bownum < e2.charoffsetF){
					if(TokenMap.get(bownum) != null)
						if(set.contains(TokenMap.get(bownum).text))
							re = "true";
					bownum++;
				}
			}

			else if(D[5].equals("abv")){
				bownum = e2.charoffsetF;
				while(bownum < Integer.parseInt(keynum)){
					if(TokenMap.get(bownum) != null)
						if(set.contains(TokenMap.get(bownum).text))
							re = "true";
					bownum++;
				}



			}else{

				bownum = e1.charoffsetR;
				while(bownum < e2.charoffsetF){
					if(TokenMap.get(bownum) != null)
						if(set.contains(TokenMap.get(bownum).text))
							re = "true";
					bownum++;
				}

			}
			return re;
		}



	private static String sec_key_asso(Entity e1, Entity e2,
			HashMap<Integer, token> TokenMap, String[] D, String keynum) {
			int bownum = 0;
			String re = "false";
			HashSet<String> set = new HashSet<String>();

			set.add("ASSOCIATE");
			set.add("ASSOCIATES");
			set.add("ASSOCOATED");
			set.add("ASSOCIATION");

			bownum = 0;

			if(D[5].equals("avb")){
				bownum = Integer.parseInt(keynum);
				while(bownum < e2.charoffsetF){
					if(TokenMap.get(bownum) != null)
						if(set.contains(TokenMap.get(bownum).text))
							re = "true";
					bownum++;
				}
			}

			else if(D[5].equals("abv")){
				bownum = e2.charoffsetF;
				while(bownum < Integer.parseInt(keynum)){
					if(TokenMap.get(bownum) != null)
						if(set.contains(TokenMap.get(bownum).text))
							re = "true";
					bownum++;
				}



			}else{

				bownum = e1.charoffsetR;
				while(bownum < e2.charoffsetF){
					if(TokenMap.get(bownum) != null)
						if(set.contains(TokenMap.get(bownum).text))
							re = "true";
					bownum++;
				}

			}
			return re;
		}



	private static String sec_key_stim(Entity e1, Entity e2,
			HashMap<Integer, token> TokenMap, String[] D, String keynum) {
			int bownum = 0;
			String re = "false";
			HashSet<String> set = new HashSet<String>();

			set.add("STIMULATE");
			set.add("STIMULATION");
			set.add("STIMULATED");
			set.add("STIMULATES");

			bownum = 0;

			if(D[5].equals("avb")){
				bownum = Integer.parseInt(keynum);
				while(bownum < e2.charoffsetF){
					if(TokenMap.get(bownum) != null)
						if(set.contains(TokenMap.get(bownum).text))
							re = "true";
					bownum++;
				}
			}

			else if(D[5].equals("abv")){
				bownum = e2.charoffsetF;
				while(bownum < Integer.parseInt(keynum)){
					if(TokenMap.get(bownum) != null)
						if(set.contains(TokenMap.get(bownum).text))
							re = "true";
					bownum++;
				}



			}else{

				bownum = e1.charoffsetR;
				while(bownum < e2.charoffsetF){
					if(TokenMap.get(bownum) != null)
						if(set.contains(TokenMap.get(bownum).text))
							re = "true";
					bownum++;
				}

			}
			return re;
		}



	private static String sec_key_regu(Entity e1, Entity e2,
			HashMap<Integer, token> TokenMap, String[] D, String keynum) {
			int bownum = 0;
			String re = "false";
			HashSet<String> set = new HashSet<String>();

			set.add("REGULATE");

			bownum = 0;

			if(D[5].equals("avb")){
				bownum = Integer.parseInt(keynum);
				while(bownum < e2.charoffsetF){
					if(TokenMap.get(bownum) != null)
						if(set.contains(TokenMap.get(bownum).text))
							re = "true";
					bownum++;
				}
			}

			else if(D[5].equals("abv")){
				bownum = e2.charoffsetF;
				while(bownum < Integer.parseInt(keynum)){
					if(TokenMap.get(bownum) != null)
						if(set.contains(TokenMap.get(bownum).text))
							re = "true";
					bownum++;
				}



			}else{

				bownum = e1.charoffsetR;
				while(bownum < e2.charoffsetF){
					if(TokenMap.get(bownum) != null)
						if(set.contains(TokenMap.get(bownum).text))
							re = "true";
					bownum++;
				}

			}
			return re;
		}



	private static String sec_key_known(Entity e1, Entity e2,
			HashMap<Integer, token> TokenMap, String[] D, String keynum) {
			int bownum = 0;
			String re = "false";
			HashSet<String> set = new HashSet<String>();

			set.add("KNOWN");

			bownum = 0;

			if(D[5].equals("avb")){
				bownum = Integer.parseInt(keynum);
				while(bownum < e2.charoffsetF){
					if(TokenMap.get(bownum) != null)
						if(set.contains(TokenMap.get(bownum).text))
							re = "true";
					bownum++;
				}
			}

			else if(D[5].equals("abv")){
				bownum = e2.charoffsetF;
				while(bownum < Integer.parseInt(keynum)){
					if(TokenMap.get(bownum) != null)
						if(set.contains(TokenMap.get(bownum).text))
							re = "true";
					bownum++;
				}



			}else{

				bownum = e1.charoffsetR;
				while(bownum < e2.charoffsetF){
					if(TokenMap.get(bownum) != null)
						if(set.contains(TokenMap.get(bownum).text))
							re = "true";
					bownum++;
				}

			}
			return re;
		}


	private static String sec_key_inter(Entity e1, Entity e2,
			HashMap<Integer, token> TokenMap, String[] D, String keynum) {
		int bownum = 0;
		String re = "false";
		HashSet<String> set = new HashSet<String>();

		set.add("INTERACTION");
		set.add("INTERACT");
		set.add("INTERACTS");
		set.add("INTEARACTED");

		bownum = 0;

		if(D[5].equals("avb")){
			bownum = Integer.parseInt(keynum);
			while(bownum < e2.charoffsetF){
				if(TokenMap.get(bownum) != null)
					if(set.contains(TokenMap.get(bownum).text))
						re = "true";
				bownum++;
			}
		}

		else if(D[5].equals("abv")){
			bownum = e2.charoffsetF;
			while(bownum < Integer.parseInt(keynum)){
				if(TokenMap.get(bownum) != null)
					if(set.contains(TokenMap.get(bownum).text))
						re = "true";
				bownum++;
			}



		}else{

			bownum = e1.charoffsetR;
			while(bownum < e2.charoffsetF){
				if(TokenMap.get(bownum) != null)
					if(set.contains(TokenMap.get(bownum).text))
						re = "true";
				bownum++;
			}

		}
		return re;
	}


	private static String sec_key_bind(Entity e1, Entity e2,
			HashMap<Integer, token> TokenMap, String[] D,String keynum) {
		int bownum = 0;
		String re = "false";
		HashSet<String> set = new HashSet<String>();

		set.add("BIND");
		set.add("BINDS");
		set.add("BINDING");

		bownum = 0;

		if(D[5].equals("avb")){
			bownum = Integer.parseInt(keynum);
			while(bownum < e2.charoffsetF){
				if(TokenMap.get(bownum) != null)
					if(set.contains(TokenMap.get(bownum).text))
						re = "true";
				bownum++;
			}
		}

		else if(D[5].equals("abv")){
			bownum = e2.charoffsetF;
			while(bownum < Integer.parseInt(keynum)){
				if(TokenMap.get(bownum) != null)
					if(set.contains(TokenMap.get(bownum).text))
						re = "true";
				bownum++;
			}



		}else{

			bownum = e1.charoffsetR;
			while(bownum < e2.charoffsetF){
				if(TokenMap.get(bownum) != null)
					if(set.contains(TokenMap.get(bownum).text))
						re = "true";
				bownum++;
			}

		}
		return re;
	}




















}




