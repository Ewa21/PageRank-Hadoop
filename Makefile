#
# This is a simple makefile to assist with quickly building the Exercises of MP2.
#
# To build and execute a particular exercise:
#    - For a single exercise, type 'make runA' to run exercise A.
#    - For all exercises, 'make'.
#
#
HADOOP_CLASSPATH := ${JAVA_HOME}/lib/tools.jar
export HADOOP_CLASSPATH

HDFS=user/lada04/bigdata

OBJDIR=build

JAR := MapReducePSO.jar

TARGETS := $(addprefix run, A B C D E F)

.PHONY: final $(TARGETS) clean

final: $(TARGETS)

runA: $(OBJDIR)/PageRank.class
	jar -cvf $(JAR) -C $(OBJDIR)/ ./
	hdfs dfs -rm -r -skipTrash -f /$(HDFS)/lab5/output1/*
	hadoop jar $(JAR) PageRank -D N=4  -D K=10  -D B=0.8  /user/adampap/wikiSmall  /$(HDFS)/lab5/output1
	@echo "Run the following command to read the output file:"
	@echo "hdfs dfs -cat /user/lada04/bigdata/lab5/output1/Final/part*"

runB: $(OBJDIR)/PageRank.class
	jar -cvf $(JAR) -C $(OBJDIR)/ ./
	hdfs dfs -rm -r -skipTrash -f /$(HDFS)/lab5/output2/*
	hadoop jar $(JAR) PageRank -D N=4  -D K=10  -D B=0.8  /user/adampap/wikiDeadEnds /$(HDFS)/lab5/output2
	@echo "Run the following command to read the output file:"
	@echo "hdfs dfs -cat /user/lada04/bigdata/lab5/output2/Final/part*"

runC: $(OBJDIR)/PageRank.class
	jar -cvf $(JAR) -C $(OBJDIR)/ ./
	hdfs dfs -rm -r -skipTrash -f /$(HDFS)/lab5/output3
	hadoop jar $(JAR) PageRank -D N=4  -D K=10  -D B=0.8  /user/adampap/wikiSpiderTraps /$(HDFS)/lab5/output3
	@echo "Run the following command to read the output file:"
	@echo "hdfs dfs -cat /user/lada04/bigdata/lab5/output3/Final/part*"

runD: $(OBJDIR)/VisitorHour.class
	jar -cvf $(JAR) -C $(OBJDIR)/ ./
	hdfs dfs -rm -r -skipTrash -f /$(HDFS)/4-output/
	hadoop jar $(JAR) VisitorHour /user/adampap/WHV /$(HDFS)/4-output
	@echo "Run the following command to read the output file:"
	@echo "hdfs dfs -cat /$(HDFS)/4-output/part*"

runE: $(OBJDIR)/TopPopularLinks.class
	jar -cvf $(JAR) -C $(OBJDIR)/ ./
	hdfs dfs -rm -r -skipTrash -f /$(HDFS)/E-output/
	hadoop jar $(JAR) TopPopularLinks -D N=5 /user/adampap/psso/links /$(HDFS)/E-output
	@echo "Run the following command to read the output file:"
	@echo "hdfs dfs -cat /$(HDFS)/E-output/part*"

runF: $(OBJDIR)/PopularityLeague.class
	jar -cvf $(JAR) -C $(OBJDIR)/ ./
	hdfs dfs -rm -r -skipTrash -f /$(HDFS)/F-output/
	hadoop jar $(JAR) PopularityLeague -D league=/$(HDFS)/misc/league.txt /user/adampap/psso/links /$(HDFS)/F-output
	@echo "Run the following command to read the output file:"
	@echo "hdfs dfs -cat /$(HDFS)/F-output/part*"


$(OBJDIR)/%.class: %.java | $(OBJDIR)
	hadoop com.sun.tools.javac.Main $< -d $(OBJDIR)

$(OBJDIR):
	mkdir $@

.PHONY: clean
clean:
	rm -f $(OBJDIR)/* $(JAR)
	hdfs dfs -rm -r -skipTrash -f /$(HDFS)/*-output/
