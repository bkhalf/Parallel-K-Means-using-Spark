package kmeans;
import java.util.ArrayList;
import java.util.List;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.ml.linalg.Vectors;

import scala.Tuple2;

public class Main {
	
	static java.util.List<Vector> centroids;

	public static void main(String[] args) {
		if(args.length != 4) {
			System.out.print("Please entre 4 arguments : <inputpath> <outpath> <number of centroids> <dimensions> .") ;
			System.exit(-1);
		}
		
		String inputPath = args[0] ;
		String outPath = args[1] ;
		int k = Integer.valueOf(args[2]) ;
		int dim = Integer.valueOf(args[3]) ;
		SparkConf conf = new SparkConf().setMaster("local[*]").setAppName("K-means App");
		JavaSparkContext sc = new JavaSparkContext(conf);

		JavaRDD<String> file  = sc.textFile(inputPath);
		JavaRDD<Vector> parsedData = file.map(line -> {
			String[] sarray = line.split(",");
			double[] values = new double[sarray.length - 1];
			for (int i = 0; i < dim ; i++) {
				values[i] = Double.parseDouble(sarray[i]);
			}
			return Vectors.dense(values);		
		}) ;
		
		centroids =  parsedData.takeSample(false,k) ;
		
		int itr = 0 , maxItr = 50;
		long start =  System.currentTimeMillis() ;
		long end;
		while(true) {
			
			JavaPairRDD<Integer,Vector> assigned = parsedData.mapToPair(d ->{
				int nearCentroid = -1 ; 
				double min = Integer.MAX_VALUE ;
				for(int i = 0 ; i < k ; i++) {
					double squares = 0 ;
					Vector v1 = centroids.get(i);
					Vector v2 = d;
					for(int j = 0 ; j < dim ; j++) {
						squares += Math.pow(v1.apply(j) - v2.apply(j), 2);
					}
					double euclideanDistance = Math.sqrt(squares) ;  
					if(euclideanDistance < min) {
						min = euclideanDistance ;
						nearCentroid = i ;
					}
				}
				return new Tuple2<Integer,Vector>(nearCentroid, d);
			});
			
			assigned.cache();

			JavaPairRDD<Integer,Tuple2<Integer,Vector>> countAndSumByKey = 
					assigned.mapToPair(t ->new Tuple2<>(t._1, new Tuple2<>(1, t._2)))
					.reduceByKey((a, s)->{
				double[] sum = new double[a._2.size()];
				int count;
				for(int i = 0 ; i < dim ; i++) {
					sum[i] = a._2.apply(i) + s._2.apply(i);
				}
				count = s._1 + a._1;
				return new Tuple2<Integer, Vector>(count, Vectors.dense(sum));
			});
			
			JavaPairRDD<Integer,Vector> newCentroids = countAndSumByKey.mapToPair(data->{
				double[] NEW = new double[dim];
				for(int i = 0 ; i < dim ; i++) {
					NEW[i] = data._2._2.apply(i) / data._2._1;
				}
				return new Tuple2<Integer, Vector>(data._1, Vectors.dense(NEW));
			});
			
			List<Vector> newCentroisdsList = newCentroids.values().collect() ;
			
			Double tolerance = 0.000001;
			Double SSE = 0.;
			for(int i=0;i<k;i++) {
				Vector v1 = centroids.get(i);
				Vector v2 = newCentroisdsList.get(i);
				for(int j = 0 ; j < dim ; j++) {
					SSE += Math.pow(v1.apply(j) - v2.apply(j), 2);
				}
			}
			
			centroids = newCentroisdsList;
			if(Math.sqrt(SSE) < tolerance || (itr+1) >= maxItr) {
				newCentroids.coalesce(1, true).sortByKey().saveAsTextFile(outPath);
				end =  System.currentTimeMillis() ;
				break ;
			}
				

				itr++ ;
		}
		sc.close();
		
		System.out.println("*****************************SUCCESSFUL OPERATION*****************************\n\n");
		System.out.println(" *CENTROIDS*\n ");
		for(int i =0; i<k ;i++) {
			System.out.println("The centriod "+i+" = "+centroids.get(i));
		}
		System.out.println("*****************************\n");
	
		System.out.println(" *PERFORMANCE*\n");
		System.out.println("*****************************\n");
		System.out.println("The total time = "+(end - start)+" ms ");
		System.out.println("The numbers of iteration token = "+itr+" iterations ");
		System.out.println("*****************************\n");
		
	}
	//Running the code
	//write on the cmd in the project folder
	//mvn clean install package
	//write in the target folder
	//java -jar kmeans-0.0.1-SNAPSHOT.jar C:\Users\pc\Desktop\kmeans\iris.txt C:\Users\pc\Desktop\kmeans\oo 3 4
}
