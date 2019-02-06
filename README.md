
## Using knn to classify 10000 testing pictures based on 60000 training pictures with Python in numpy 
>### training and testing files are available in http://yann.lecun.com/exdb/mnist/
>### require packages: 
>>> * struct ( unpack the resource files)
>>> * numpy ( calculating distance and sort the neighbors)
>>> * time ( find out the actual "wall clock" time that it took in KNN)
>>> * pickle ( store the result)


>### The code also calculates real time costs for each part and uses package pickle to store the sorted neighbors and distance matrix
 
>### The results include accuracy rate and confusion matrix



# MNIST WITH KNN
> 1. Using knn to classify 8400 testing pictures based on 33600 training pictures with Python in numpy. 
> 2. Using knn to classify 28000 testing picture based on 42000 training pictures and submit the result on kaggle.
Training and testing files are available in http://yann.lecun.com/exdb/mnist/



One to two paragraph statement about your product and what it does.

![](header.png)

## Installation

OS X & Linux:

```sh
npm install my-crazy-module --save
```

Windows:

```sh
edit autoexec.bat
```

## Usage example

A few motivating and useful examples of how your product can be used. Spice this up with code blocks and potentially more screenshots.

_For more examples and usage, please refer to the [Wiki][wiki]._

## Development setup

Describe how to install all development dependencies and how to run an automated test-suite of some kind. Potentially do this for multiple platforms.

```sh
make install
npm test
```

## Release History

* 0.2.1
    * CHANGE: Update docs (module code remains unchanged)
* 0.2.0
    * CHANGE: Remove `setDefaultXYZ()`
    * ADD: Add `init()`
* 0.1.1
    * FIX: Crash when calling `baz()` (Thanks @GenerousContributorName!)
* 0.1.0
    * The first proper release
    * CHANGE: Rename `foo()` to `bar()`
* 0.0.1
    * Work in progress

## Meta

Your Name – [@YourTwitter](https://twitter.com/dbader_org) – YourEmail@example.com

Distributed under the XYZ license. See ``LICENSE`` for more information.

[https://github.com/yourname/github-link](https://github.com/dbader/)

## Contributing

1. Fork it (<https://github.com/yourname/yourproject/fork>)
2. Create your feature branch (`git checkout -b feature/fooBar`)
3. Commit your changes (`git commit -am 'Add some fooBar'`)
4. Push to the branch (`git push origin feature/fooBar`)
5. Create a new Pull Request

<!-- Markdown link & img dfn's -->
[npm-image]: https://img.shields.io/npm/v/datadog-metrics.svg?style=flat-square
[npm-url]: https://npmjs.org/package/datadog-metrics
[npm-downloads]: https://img.shields.io/npm/dm/datadog-metrics.svg?style=flat-square
[travis-image]: https://img.shields.io/travis/dbader/node-datadog-metrics/master.svg?style=flat-square
[travis-url]: https://travis-ci.org/dbader/node-datadog-metrics
[wiki]: https://github.com/yourname/yourproject/wiki
