# The Phishing Email Was Actually From Google, but Should I Trust It?

Last night I received one of the more interesting phishing attacks I have seen personally. I work as an AI researcher at a cybersecurity company, so after a few steps I became more interested in studying the attack than following the caller's instructions. Still, I can easily see how this could succeed against someone who is distracted, worried about losing an important Gmail account, or simply trying to cooperate with what appears to be Google's security team.

The most interesting part was that several things I saw on my phone were genuinely produced by Google. The account recovery prompt was real. The verification request was real. The later security email also came through Google's infrastructure.

The attacker was exploiting the meaning I would assign to those legitimate messages.

I would describe this technique as **trusted-channel laundering**. An attacker deliberately triggers legitimate security mechanisms from a trusted service, then uses a phone call or another social channel to provide a false explanation of what those mechanisms mean. Instead of constructing a convincing imitation of Google, the attacker gets Google to produce much of the convincing material.

![Fake Google Call](/images/fake_google_call.jpg)

There is an important defense that makes this particular attack much easier to stop. Google's own current security guidance says very clearly:

> “Google will never call you about your account security.”

Google also says it will never call and ask you to read a verification code or approve a device prompt. Even messages from legitimate Google domains should not be interpreted as proof that a caller represents Google.

That simple rule turns out to be extremely relevant to what happened.

## Step 1: “Google” calls me

At around 8 PM Pacific time, I received a call from:

**650-215-XXXX**

The caller ID displayed **Google**.

The person said that someone had recently changed the contact phone number associated with my Gmail account. He told me that Google's Trust and Safety team would contact me shortly to help secure the account.

A few minutes later, another call arrived from:

**347-329-XXXX**

This time a man speaking fluent English said he worked for Google's security team. He explained that somebody had changed the contact information on my Gmail account and that he needed to verify my identity and help me recover it.

I am including both phone numbers because they were the numbers displayed during this incident and may be useful to someone searching for the same scam.

They should not be treated as durable identities for the attackers.

Caller ID names and numbers can be spoofed. A scammer can make a call appear to originate from a different number and can manipulate the displayed caller name. Even if these particular numbers were controlled by the attackers during my call, they can switch numbers at any time. Carriers may block them, Google may take action if its name is being abused, and the numbers themselves may later be reassigned.

So the useful indicator here is:

**These were the numbers used in this incident.**

The stronger security signal is the behavior that followed.

The timing also felt strange. It was around 8 PM in California and 11 PM in New York. More importantly, I had never asked Google for support.

### Why did they use two calls?

The first call established the story before the second caller asked me to do anything sensitive.

By the time the supposed Google security engineer appeared, I had already been told that there was an account incident and that another Google employee would contact me.

The second call therefore arrived as an expected event.

That sequencing matters. The attacker was creating context first, then asking for authorization.

## Step 2: They made Google send me a real recovery prompt

While “Google Security” was talking to me, my Gmail app displayed a Google account recovery notification.

A device in New York was attempting to recover my account.

That was a real Google prompt.

The caller immediately explained that this was expected because he was working on my recovery case. He told me to click **Yes** because the New York device belonged to the Google security process he had just initiated.

This is where the attack became much more interesting.

Google prompts can be used during account recovery. If Google sees an unusual sign-in or recovery attempt, the prompt can contain information such as the device and approximate location so that the account owner can reject an attempt they did not initiate.

The attacker had triggered exactly the warning that was supposed to protect me, then used the phone call to reinterpret it.

The notification was effectively asking:

> Someone in New York is trying to recover this account. Is that you?

The caller's explanation was:

> Yes, that is me in New York helping you recover your account.

The same event now had two possible meanings, and the security outcome depended on which explanation I believed.

### Why did they use Google's own recovery prompt?

Because a genuine Google notification carries much more credibility than a fake webpage or a phishing email.

Once the victim accepts the premise that the person on the phone represents Google, suspicious information inside the notification can start working in the attacker's favor.

The location **New York** could have warned me that somebody elsewhere was trying to access my account.

Under the attacker's explanation, it became evidence that the supposed Google security engineer was doing exactly what he had promised.

That is the core pattern of this attack.

## Step 3: “Please enter my job ID, 48”

Then came my favorite part of the attack.

The caller told me that his **Google employee ID was 48** and instructed me to use `48` in the authentication prompt.

Of course, `48` was functioning as part of Google's authentication challenge.

Google can use number matching as part of an authentication flow. A number displayed during one part of the transaction must correspond to the trusted device authorizing it.

The attacker gave the number a new meaning.

The security system was effectively saying:

> Verify that this authentication session is the one you intended to authorize.

The caller told me:

> This is my Google employee ID. Enter 48 to verify that I am helping you.

That is a subtle and effective semantic substitution.

For the victim, typing `48` feels almost harmless. There is no password being disclosed, no long OTP being read aloud, and no suspicious website being opened.

It sounds like entering an employee identifier into Google's own application.

### Why did they call 48 an “employee ID”?

People have been trained for years not to share passwords or verification codes.

A two-digit employee ID feels very different.

The attacker did not need to hide the authentication token. He changed the victim's understanding of what the token represented.

This is one of the most interesting parts of the attack from a security-design perspective.

I refused to enter it.

## Step 4: When the first path failed, they pivoted

After I questioned the first request, the caller immediately changed approaches.

An unfamiliar Gmail account:

**[{something with letters and numbers}@gmail.com](mailto:{something with letters and numbers}@gmail.com)**

attempted to establish a recovery relationship involving my email address. Google then sent me another legitimate request related to that recovery process.

There was another verification code involved, and the caller wanted me to provide or confirm it.

Again, I refused.

I am including the Gmail address because it was directly involved in this incident and may help other people identify the same campaign.

Like the phone numbers, however, this email address should be treated as a temporary incident indicator rather than a permanent attacker identity.

The scammer can create another Gmail account at any time. Google may also suspend or remove the account after receiving abuse reports. By the time someone reads this article, **[{something with letters and numbers}@gmail.com](mailto:{something with letters and numbers}@gmail.com)** may already be inactive, or the same group may be using completely different accounts.

The behavior is more durable than the identifier.

### Why did they switch to another Gmail account?

The first route depended on getting me to authorize recovery of my own account.

Once I refused, the attacker needed another Google workflow capable of generating trusted messages to my inbox.

A recovery-email relationship provides exactly that.

The attacker can initiate an action involving an attacker-controlled Google account, cause Google to send the victim a legitimate message, then explain that message over the phone as part of the victim's supposed security incident.

This also showed that the person on the phone was not committed to one carefully scripted exploit. He appeared to have several Google account workflows available and was choosing among them interactively based on what I accepted or rejected.

## Step 5: A real Google security email appears to confirm “James Wilson”

The next step was the most convincing.

A security alert associated with the unfamiliar account was copied to my email address. The alert concerned an:

**“App password created to sign in to your account”**

event.

Inside the message was text along the lines of:

> If you didn't generate this password for [my email], then please proceed with the current call with Google security agent, {a common US male name}, Case ID: {some number}. More than likely, someone might be using your account.

At almost exactly the same time, the caller, who had introduced himself as {a common US male name}, told me that he had escalated my security case.

He repeated:

**Case ID: 37518**

This was a much stronger social-engineering artifact than an ordinary phishing email.

From the victim's perspective, two apparently independent channels now agree.

The person on the phone says:

**{a common US male name}, case {some number}.**

A security email delivered through Google's infrastructure also contains:

**{a common US male name}, case {some number}.**

The natural interpretation is that Google has authenticated the caller.

That conclusion does not follow.

I have not independently reproduced exactly which attacker-controlled Google field allowed that particular text to appear in the security notification, so I would treat that implementation detail as an open question. It may involve an account-controlled label, application name, metadata field, or another value that Google inserts into a legitimate notification.

The observable security problem does not depend on the exact field.

The attacker found a way to cause legitimate Google infrastructure to deliver a message containing information coordinated with the story being told on the phone.

### Why did they do this?

Because agreement between independent channels normally increases trust.

If somebody gives you a case number over the phone, and then an apparently independent email from Google contains exactly the same case number, that correlation feels difficult to fake.

The attacker was manufacturing the correlation.

He controlled the action that caused Google's system to generate the second piece of evidence.

This is where the term **trusted-channel laundering** becomes useful.

## Trusted-channel laundering

Traditional phishing often requires an attacker to imitate something trusted.

They may create a fake login page, register a confusing domain, spoof a support email, or reproduce a company's branding.

This attack has a different structure.

The attacker initiates an action inside a legitimate service.

Google then produces a genuine security artifact.

The attacker remains on another communication channel and explains what the artifact supposedly means.

The flow looks roughly like this:

**Attacker action → legitimate Google workflow → genuine Google notification → attacker-provided interpretation**

The trusted service supplies authenticity to the artifact.

The attacker supplies the semantics.

The victim sees the two together.

That is what I mean by **trusted-channel laundering**.

The attacker gets some of the credibility of Google's infrastructure transferred to a story that Google itself never authenticated.

## The deeper problem is semantic binding

From a security research perspective, the most interesting issue is what I would call a **semantic binding failure**.

An authentication system can accurately record that a trusted device approved an action.

The harder question is whether the human operating that device understood the action in the same way the security protocol intended.

During this call, the attacker repeatedly tried to replace the protocol's meaning with his own.

An account recovery attempt became:

> Google Security is helping recover your account.

A number-matching challenge became:

> Enter my employee job ID.

A recovery-email verification became:

> Verify your security case.

A security notification associated with another account became:

> Google has confirmed your security representative.

The security mechanisms can all execute correctly while the human authorizes something based on a false mental model.

This is a useful reminder that authentication involves both protocol integrity and user interpretation.

## Then I asked for his Google LDAP

At this point, I decided to test whether the caller could handle something outside his script.

I asked:

> “What is your Google LDAP?”

He hung up.

No explanation. No argument. The call simply ended.

For this particular scammer, **LDAP** turned out to be a rather effective magic word.

I would not treat this as a real defense technique.

The next scammer can learn the terminology. They can prepare a believable LDAP identifier, invent one confidently, or develop a response explaining why they cannot provide it.

The interesting part was that the question moved the interaction away from the protocol the attacker had prepared.

Until that moment, he controlled what happened next. He initiated a Google workflow, knew what I was about to see, and immediately supplied an explanation for it.

An unexpected question reversed that asymmetry for a few seconds.

For someone analyzing the scam, that was useful.

For a normal user, there is a much easier defense: hang up.

## Google has a help page that describes this exact class of scam

Afterward, I looked through Google's security documentation.

Google has a help page specifically about **Google Account Security Scam via Phone Call**.

The most useful sentence is also the simplest:

> **“Google will never call you about your account security.”**

Google additionally warns users that scammers may send messages from legitimate Google domains to make their claims appear authentic.

That warning is particularly important here.

A message can genuinely originate from Google's infrastructure because the attacker triggered a legitimate Google workflow.

The origin of the email tells you who delivered the message.

It does not automatically establish who initiated the underlying event, what that event means for your account, or whether the person currently talking to you represents Google.

This is exactly why checking only the sender domain would have been insufficient during this attack.

## What I think Google could strengthen

Google's security mechanisms gave me enough information to stop the attack. I still think this incident exposes several places where security UX could make the attacker's semantic manipulation harder.

### 1. Make user intent explicit in recovery prompts

Instead of only asking:

**Are you trying to recover your account?**

the prompt could emphasize:

**Did you personally initiate this account recovery process on another device?**

It could also include:

**Google employees will never ask you to approve this prompt during a phone call.**

The goal is to bind the approval to an action the user remembers initiating.

### 2. Explain where number matching values should come from

The weakness of `48` was not simply that the number was short.

The problem was that a caller could plausibly assign another meaning to it.

A stronger prompt might say:

**Select this number only if it matches the number displayed on the device where you personally initiated sign-in. Never select a number given to you by someone on a phone call.**

This tells the user about the expected provenance of the number.

### 3. Treat new recovery relationships carefully

If my email has suddenly become associated with another Google account as a recovery address, subsequent security messages about that account should make the relationship very clear.

Something like:

**This security alert concerns another Google account that listed your email as a recovery address. It does not mean that your own Google Account has been compromised.**

That one sentence would make this attack significantly harder.

### 4. Visually isolate attacker-controlled text

Any user-controlled field inserted into a security email should be considered potentially adversarial.

Account names, app labels, device names, recovery information, project names, and similar metadata should be visually separated from Google's own authoritative security instructions.

Otherwise an attacker may be able to place social-engineering text inside a message that inherits the visual authority of a Google security alert.

### 5. Think beyond device possession toward intent verification

Modern authentication systems are increasingly good at confirming that a trusted device participated in an authorization.

Social engineering creates another question:

**Did the user understand exactly what operation they were authorizing and who initiated it?**

That is a much harder security problem.

This incident provides a clean example because the attacker's main tool was repeatedly changing the perceived meaning of legitimate authentication events.

## What should a user actually do?

You do not need to understand trusted-channel laundering, inspect email headers, know Google's internal terminology, or ask anyone for their LDAP.

Google provides a much simpler rule:

**Google will never call you about your account security.**

If somebody claiming to be Google Account Security calls you unexpectedly, end the call.

Then open your Google Account security page yourself. Navigate there independently through Google's app or website and inspect recent security activity.

If an authentication prompt appears that you did not personally initiate, reject it.

Do not read verification codes to someone on the phone.

Do not type a number into a security prompt because a caller tells you that it is their employee ID, case number, job number, or anything else.

If an email from Google arrives while the person is talking to you, inspect the event independently. The fact that Google genuinely delivered the email does not authenticate the caller.

And if you encounter the specific indicators I saw:

* **650-215-5242**
* **347-329-3627**
* **[fredwright726@gmail.com](mailto:fredwright726@gmail.com)**
* **“James Wilson”**
* **Case ID 37518**
* **“job ID 48”**

they may help connect your incident to the same scam campaign.

Do not rely on them as a blocklist.

Scammers can change phone numbers, caller ID names, email accounts, employee names, case numbers, and scripts very quickly. Google or telephone carriers may also disable some of these identifiers after abuse is reported.

The reusable defense is to recognize the mechanism.

What made this incident interesting to me as both an AI researcher and someone working in security was how little fake infrastructure the attacker actually needed. He could trigger real Google security workflows, observe what I would see next, and provide a carefully prepared interpretation over the phone.

For a few minutes, Google itself was producing many of the artifacts that made his story believable.

That is why a legitimate security message still deserves one more question:

**Who caused this message to be sent, and what action am I actually being asked to authorize?**
