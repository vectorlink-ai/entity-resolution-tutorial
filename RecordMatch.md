# Leveraging LLMs and Machine Learning for Record Matching

When currating data sets which relate to *entities*, whether these be products, people or corporations, we encounter a problem in knowning when one thing is the same as another. This process is variously known as record matching, record linking and entity resolution.

In *the real world*, determination of when two things are the same sometimes has a very definitive answer. Joe Bloggs may refer to more than one individual, but each of these individuals is unique and unambiguous. When there is a record it should be possible to assign to which of these Joe's they are referring, at least in principle. Sometimes however, the designation of an entity is more complicated and requires additional logic, for instance when a corporation has subsidiaries. In our discussion we are going to restrict ourselves to the former problem, where the real identity of something has a clear notion.

Record matching has been around for a very long time, and therefore has lots of well known approaches. But LLMs ([Large Language Models]([LLM](https://en.wikipedia.org/wiki/Large_language_model))) have enabled some new and surprisingly effective techniques that are often much more robust. Content which would have been difficult or even impossible to match on previously can now be utilised.

## Practical matching with LLMs

To demonstrate how LLMs can be leveraged for record matching we are going to use an LLM only solution. In practice, a mixture of approaches improves results, where semantically rich and textual content utilises the LLM and various quantitative, highly structure or temporal properties are matched with other approaches. Yet the pure LLM approach is so effective that for many use-cases we can ignore the more complex solution.

The general strategy is as follows:

- Obtain vector embeddings using fields from our records.
  - Fields can be individual or composite (e.g. first-middle-last name, full address)
  - We vectorise one domain for matching ourselves, or two domains for cross-matching
- Obtain a small training set to help us discover a classifier.
- Choose a particular vectorised field or combination to obtain candidate records (also known as a filtering approach).
- Use the filter to find candidates and the classifier to obtain matches.

Not only does this approach make use of the advanced AI capacities of LLMs, it leverages AI to avoid the quadratic-explosion of possible combinations that is lurking in the matching problem if we have to compare each record against every other.

In this example, we're going to use the ACM / DBLP data-set which matches records across two databases of paper publications. This has the advantage of having a small known sample of exact matches which we can use to train. In many real world cases you will not have such a training set easily avaialble, but we'll discuss how you can obtain a similar dataset for your problem when we get to training.

## Vectorizing

Vectorisation is the process of obtaining high-dimensional vectors from (in this case) textual content. For this experiment we will use OpenAI and we will obtain the embeddings using an open-source rust tool we have built called VectorLink.

The ACM and DBLP datasets can be found [here](http://github.com/terminusdb-labs/acm-dblp).

First, take the two files and vectorize both:

```shell
cargo run --release --bin generate-vectors csv-columns --config ~/dev/entity-resolution/config -i ~/dev/entity-resolution/DBLP2.utf8.csv -o ~/tmp/dblp/ --template-dir ~/tmp/dblp/templates --id-field "id"
```

And the second one...

```shell
cargo run --release --bin generate-vectors csv-columns --config ~/dev/entity-resolution/config -i ~/dev/entity-resolution/ACM.csv -o ~/tmp/acm/ --template-dir ~/tmp/acm/templates --id-field "id"
```

## Indexing

```shell
cargo run --release --bin index-graph-field -- --graph-directory ~/tmp/acm --field record
```

## Guessing weights

```shell
cargo run --release -- find-weights -s ~/tmp/dblp -t ~/tmp/acm --id-field "id" -f "record" --initial-threshold 0.3 --config ~/tmp/dblp/config --answers-file ~/dev/entity-resolution/DBLP-ACM_perfectMapping.csv --comparison-fields "title" "authors" "venue" "year"
```

## Find weights

```shell
cargo run --release --bin generate-vectors -- find-weights --config ~/tmp/dblp/config -s ~/tmp/dblp -t ~/tmp/acm -f "record" --initial-threshold 0.3 --answers-file ~/dev/entity-resolution/DBLP-ACM_perfectMapping.csv --comparison-fields "title" "authors" "venue" "year"
```

We find the following weights:

```json
{ "year":          -9.939934,
  "title":        -25.451017,
  "__INTERCEPT__":  3.1453862,
  "authors":      -21.432272,
  "venue":         -1.2297093 }
```

## Run match search
