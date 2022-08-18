# Zero hacking problem: do we really protect the customers?

Disclaimer: Nothing in this blog is related to the author's day-to-day work. The content is neither affiliated nor sponsored by any companies.

A few weeks ago, Jason came to me with a question from his product manager buddy:

> How can we tell the effectiveness of our product if a customer used our cloud security solution to protect 100 cloud machines and saw no hacking?

If my dear readers don't know Jason, please refer to [the previous blog post](https://toooold.com/2022/07/25/measure_unmeasureable.html).

![One hundred machines](/images/one-hundred.001.png)

"So, if we don't respond, the customer will stop renewing the protection subscription?" I said. Jason was virtually nodding on the screen.

Surely, business is business, so we must respond to this question. We could respond in this way: we knew the malware infection or hacking rate on the cloud was 5% before we implemented our security solution, and we reduced it to 1% after we implemented it.


"But the customer saw zero hacking events, and I mean **ZERO** in every way, even before using our product. So, how do we persuade them?" Jason said.

Let's toss coins and understand the observation dilemma. If I tossed a fair coin for 100 times and see all heads no tails, the probability can be $\frac{1}{2^{100}}=7.9\times10^{-31}$. If I tossed a double-headed coin for 100 times and see all heads no tails too, the probability can be, surely, $1$. So, **if I didn't know which coin I picked** and I just tossed 100 times and saw all heads, which coin did I choose, the fair one or the double-headed one?

"It sounds like the double-headed coin." Jason said. "However, it's still possible that you chose the fair one; it's just very unlikely."

Yes, and Jason's intuition can be quantified in the probability way.

If we know the malware or hacking infection rate is 5% and no hacking occurred on 100 independent machines, the probability was $0.95^{100}=0.6\%$. With our security solution, the malware or hacking rate was reduced to $1\%$, and we saw no hacking, so the probability became $0.99^{100}=36.6\%$. We increased the likelihood of none of these 100 machines being hacked from $0.6\%$ to $36.6\%$!

"Please wait; I understand probability as well. We can't convince the customer that $36.6\%$ is significantly better than $0.6\%$ because we saw no hacking events in the first place." Jason stated.

Jason, you were correct. We had observations and had to state how confident we were by having the range of assumed adverse event rates leading to a probability of $5\%$ or more as a $95\%$ confidence interval.

> "Adverse event" meant that even after we applied the protection, the machine was still hacked. What a stroke of bad luck.

When referring to [confidence interval](https://en.wikipedia.org/wiki/Confidence_interval), like $95\%$ confidence interval for most statistical problems, we use this equation for binomial distribution:

$$
\widehat{p}\pm Z_{1-\alpha/2}\sqrt{\frac{p(1-p)}{n}}
$$

But the zero observation case didn't satisfy its underlining assumption of normal distribution. Fortunately, we could get back to the original definition of confidence interval. To get $95\%$ confidence interval, we will need:

$$
(1-p)^n=0.05
$$

where $p$ is the assumed probability of "bad luck" adverse events. Thus, we have

$$
n\ln(1-p)=\ln(0.05)
$$

Since $\ln(1-p)\sim-p$ using Taylor expansion for small $p$ value ("Yes, I remember it from my information theory class!" Jason said.), we have the upper bound of $95\%$ confidence level as

$$
p=\frac{3}{n}
$$

For $n=100$ in our case, $95\%$ confidence level required $p$ from $0$ to $0.03$ which meant it could tolerant $0\%$ to $3\%$ malware or hacking rate where we observed zero hacking event for 100 independent machines. If the customer didn't have our security solution, $p$ was $5\%$ which was out of the boundary, very dangerous, and our security solution successfully reduced it to $1\%$ and it was within the $95\%$ confidence level boundary.

"So the customer should renew the product!" Jason said.

"But we have to clarify that the 100 machines must be independent because ...... wait, get back to the meeting! Jason, can you hear me?" I said.

## Reference

* Confidence interval with zero events <http://www.pmean.com/01/zeroevents.html>