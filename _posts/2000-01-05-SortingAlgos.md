---
layout: post
title: "Sorting Algos"
subtitle: "Algos of some sort. This isn't very interesting. Not the article I'm the proudest about. Need to up my game."
categories: misc
---

I never wrote them down. So I'm typing them out here. I also tend to forget about those so I'll just use this to refresh my memory 10 minutes before my next pair-coding interview. I'll also do a bit of complexity comparison between these algos, see if there's anything unexpected.

There are 5 main sorting algorithms more ML/SWE/Quant compagnies will expect any candidate at any level to understand. I'll provide an intuitive explanation about each algorithm and give a code snippet (in Python, of course — you thought I was a dev?). This will still be somewhat high-level and shouldn't be taken as a reference for CS guys trying to become SWEs, as the provided code is not meant to be computationally optimal but rather intuitive.



---

## QuickSort

Quicksort is a *Divide and Conquer* algorithm, breaking down the problem into smaller sub-problems. The process is simple:
- Pick a pivot: this is an element in the array
- Partition the array: split the array into two sub-arrays, around the pivot. Elements smaller than the pivot should be in the left sub-array, elements larger in the right sub-array.
- Recursively call the same function on the two subarrays until there's only one element in the sub-array.


<pre><code class="language-python">def quick_sort(array):

    if len(array) <=1:
        # Base Case: if there's only 1 element in the array, it is sorted. 
        # We can return it as is
        return array
    
    # Select a pivot. We take the middle element here
    pivot_index = len(array)//2
    pivot = array[pivot_index]
    # take all the elements that are equal to the pivot
    pivots = [x for x in array if x==pivot]
    

    # Partition the array
    smaller = [x for x in a if x<pivot]
    larger = [x for x in a if x>pivot]
    
    # Recursively call
    return quicksort(smaller) + pivots + quicksort(larger)
</code></pre>


***Time Complexity***:\
The average runtime complexity of this algorithm is O($n$ log $n$). 
At each call, it takes O($n$) to traverse the array to select the pivots, O($n$) to fill each of the subarray.
Ideally, the pivot is the median element, which will make the two partition function divide the arary into two halves. So there will be  O($n$ log $n$) total calls.

However, as the pivot is not guaranteed to the be the median, the sorting could be very slow and can have O($n^2$) worst-case complexity.

***Space Complexity***:\
O(log $n$).


---

## BubbleSort

BubbleSort is arguably the simplest algorithm. 

- We start at the begining of the array, and swap the first two elements if the first is greater than the second. We then go up to the next pair, until we do a full pass of the array. After the first pass, the maximum element goes to the end.
- We start again from the first element. After $k$ passes, the $k$ largest elements must have been moved to the last $k$ positions

<pre><code class="language-python">def bubble_sort(array):

    n = len(array)

    if n<=1:
        return array
    
    # Traverse through the entire array
    for i in range(n):

        # The last i elements should already be in place
        for j in range(i, n-1-i):

            # Swap the elements if the first is larger
            if array[i]>array[j]:
                array[i], array[j] = array[j], array[i]
                
    return array
</code></pre>


***Time Complexity***:\
O($n^2$), which makes it very slow for large datasets. We must traverse the array $n$ times and make $n-i-1$ (still O($n$)) comparisons per traversal. If the array is already sorted, its complexity reduces to O($n$).

***Space Complexity***:\
O(1). We modify the array in place so it doesn't require any additional memory space.

---

## InsertionSort

InsertionSort is another simple algorithm that iteratively sorts the array. It works by separating the array into a sorted one (which corresponds only to the first element at the start) and iteratively adding elements from the unsorted sub-array (on the right of the sorted sub-array) into the sorted one, at its correct position, to increase the size of the sorted sub-array by 1.

- Compare the second element with the first, and swap them if the second is smaller
- Move to the third element, compare it to the first two elements and swap them if needed
- Repeat until sorted


<pre><code class="language-python">def insertion_sort(a):
    
    n = len(array)

    if n<=1:
        return array
    
    # Traverse through the entire array
    for i in range(n):

        # Loop over all previous elements
        # And swap them if the order is wrong
        j=i-1
        while j>=0 and array[i]<array[j]:
            array[i], array[j] = array[j], array[i]
            i-=1
            j-=1

    return array
</code></pre>

***Time Complexity***:\
O($n^2$), which makes it very slow for large datasets. We must traverse the array $n$ times and make $i-1$ (still O($n$)) comparisons per traversal.

***Space Complexity***:\
O(1). We modify the array in place so it doesn't require any additional memory space.

One must note that InsertionSort is generally less efficient than other sorting algorithms and is only used when the list is small or nearly sorted. If the array is already sorted, its complexity reduces to O($n$).


---

## MergeSort

MergeSort is one of the most popular and elegant algorithms, known for its efficiency and stability. Similarly to QuickSort, it follows a *Divide and Conquer* approach, where the array is iteratively divided into two halves which are recursively sorted and merged back together.

- Divide: divide the array into two halves
- Conquer: sort each subarray individually
- Merge: Sorted subarrays are merged back in sorted order

<pre><code class="language-python">def merge(left,right):
    # This is the merge funtion
    # It merges two already sorted arrays to produce a sorted output

    # If one array is empty, we can just return the other
    if left==[]:
        return right
    if right ==[]:
        return left

    # We compare the first elements of each array
    # Add the smallest one to the final merged array 
    # And continue merging the two remaining sub-arrays
    if left[0]<=right[0]:
        return [left[0]] + merge(left[1:], right)
    if left[0]>right[0]:
        return [right[0]] + merge(left, right[1:])


def merge_sort(array):
    n = len(array)

    if n<=1:
        return array

    # Take a pivot element to split the array
    mid = len(a)//2

    # Recusively call the function on each sub-array
    left = merge_sort(a[:mid])
    right = merge_sort(a[mid:])    

    # Merge the two sub-arrays
    return merge(left, right)
</code></pre>
    

***Time Complexity***:\
O($n$ log $n$), making it more efficient than BubbleSort and InsertionSort, and as efficient as QuickSort.
The merge function has O($n$) complexity as it has two traverse linearly both sub-arrays to merge them. The splitting in half and recursive procedure is O(log $n$ ), as usual. While having the same average complexity as QuickSort, it has O($n$ log $n$) worst-case complexity, as we are guaranteed to divide the array exactly in half at each recursion, while for QuickSort it depends on if the pivot element is close to the median.



***Space Complexity***:\
O($n$), as additional space is required for he temporary used during merging. QuickSort is generally preferred for large datasets as it works in place and only requires  O(log $n$) memory.


--- 

## HeapSort

HeapSort is an algorithm based on the ***Binary Heap*** data structure. 

A ***Binary Heap*** is a binary tree with two constraints: it must be *complete*, meaning that all levels (except possibly the last one) are fully filled (and the last one is filled from left to right); and has the *Heap property*, where the key stored in each node must be greater than or equal (or lesser than or equal, depending on whether we work with a min-heap or a max-heap) than the keys in its children nodes.\
It is a popular data structure as its two main operations, *Insert* (adding an element) and *Extract* (removing the top element, either the largest or the smallest depending on whether we work with a min-heap or max-heap) have both O(log($n$)) time complexity. This includes, of course, the re-arrangement of the *Heap property* (through an algorithm called ***Heapify***).\
Actually, the Binary Heap was invented for the HeapSort algorithm. We truly are blessed.

- First convert the array into a Max Heap using Heapify
- Extract the root note (largest element), and replace it with the last node, and heaipfy again
- Repeat until the heap has only one element


So the magic of this algorithm is really the ***Heapify*** procedure. Intuitively, every node has two children, and heapify works by pushing a node downward until it satisfies the heap property (i.e. it is at least as large as both of its children). Since the heap is a complete binary tree, each time a node moves down one level, the number of nodes in that level doubles, but the height of the tree is only O(log $n$). A comparison between a parent and its children is O($1$), and in the worst case heapify keeps moving the element down level by level, giving an overall cost of O(log $n$) for a single heapify call.


<pre><code class="language-python">def heapify(array, n, i):
    
    # Initialize at root
    largest = i

    # Index of left child
    left = 2 * i + 1

    # Index of right child
    right = 2 * i + 2
    
    # If the left child is larger than the root
    if left < n and array[left] > array[largest]:
        largest = left
        
    # If the right child is larger than the root
    if right < n and array[right] > array[largest]:
        largest = right
        
    # If the largest is not root
    if largest != i:
        array[i], array[largest] = array[largest], array[i]

        #Recusively heapfiy the sub-tree
        heapify(arr, n, largest)

def heap_sort(array):
    n = len(array)

    if n<=1:
        return array
    
    # Build max heap
    for i in range(n // 2 - 1, -1, -1):
        heapify(array, n, i)
    
    # Extract elements from heap one by one
    for i in range(n - 1, 0, -1):

        # Move current root to end
        array[i], array[0] = array[0], array[i]  

        # Heapify the reduced heap
        heapify(array, i, 0)
    
    return array
</code></pre>


***Time Complexity***:\
O($n$ log $n$) in all cases. Each Heapify calls has O(log $n$) runtime, and we call it everytime we extract the largest element, which we do $n$ times. This makes it quite suitable for large datasets.

***Space Complexity***:\
O(log $n$), due to the recursive call stack. 


---

I should probably add a section about Radix Sort or Counting sort. I'll do it. But later.

---
## References

**Cracking the Coding Interview** (2008)\
Gayle Laakmann McDowell\
[Book](https://www.crackingthecodinginterview.com/){:target="_blank"}

**GeeksforGeeks**\
[Website](https://www.geeksforgeeks.org/dsa/sorting-algorithms/){:target="_blank"}

