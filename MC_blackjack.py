# Monte-Carlo Predeiction of Blackjack
# based on Sutton & Barto pp91-92
# 10-12-20
# Chris Hicks
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

# Defines a standard 52-card deck 
class standard_52_deck:

	def __init__(self):
		self.deck = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10]*4).flatten()

	# Returns a random card from the deck (with replacement)
	def infinite_random_card(self):
		return np.random.choice(self.deck)


# Keep track of a blackjack hand and returns it's value
class blackjack_hand:

	def __init__(self, c1, c2):
		self.cards = [c1, c2]
		self.usable_ace = 0

	# Count hand. Count ace as 11 if doing so wont cause bust.
	def count(self):
		total = 0
		aces = 0
		usable_ace = 0

		for card in self.cards:
			if card == 1:	# Put aces to one side and count them afterwards
				aces += 1
			else:
				total += card

		while aces > 0:
			if total <= 10 and (total+11)<=21:	# Ace is usable
				total += 11
				self.usable_ace = 1
			else:
				total += 1
			aces -= 1

		return total	

	# Add new card to hand
	def add(self, card):
		self.cards += [card]

	# If usable ace counted
	def get_usable_ace(self):
		self.count()
		return self.usable_ace
	

# Blackjack state definition
class blackjack_state:
	def __init__(self, current_sum, dealer_show, usable_ace):
		self.current_sum = int(current_sum)
		self.dealer_show = int(dealer_show)
		self.usable_ace = int(usable_ace)

	def to_string(self):
		return "current sum: {}, dealer's showing card: {}, usable ace: {}".format(self.current_sum, self.dealer_show, self.usable_ace)


# A game of blackjack
class blackjack_game:

	# Rewards of +1, -1 and 0 are given for winning, losing and drawing a game of Blackjack.
	reward = {'win':1, 'lose':-1, 'draw': 0}

	# Initialise a game of blackjack
	def __init__(self, deck):
		self.deck = deck
		self.states = []

	# One episode of blackjack with *fixed dealer and player stragegies* (fixed policy and environment).
	# Dealer strategy: (stick) on any sum 17 or greater, (hits) otherwise.
	# Policy: Player sticks if sum is 20 or 21, otherwise hits
	def play(self):

		# The game begins with two cards dealt to each the user and the dealer. 
		# Here we use the U.S. blackjack method where only the dealer's first card is face up to begin.
		d1, d2 = self.deck.infinite_random_card(), self.deck.infinite_random_card()
		p1, p2 = self.deck.infinite_random_card(), self.deck.infinite_random_card()
		dealer_hand = blackjack_hand(d1, d2)
		player_hand = blackjack_hand(p1, p2)
		self.states += [blackjack_state(player_hand.count(), d1, player_hand.get_usable_ace())]

		# If the player has 21 immediately then she wins unless dealer also (natural)
		if player_hand.count() == 21:
			if dealer_hand.count() == 21:
				return self.reward['draw'], self.states
			else:
				return self.reward['win'], self.states

		# If player does not have natural, can request additional cards (hits) until (goes bust) or chooses to (stick)
		
		# Player always hits if score < 20:
		while player_hand.count() < 20:
			player_hand.add(self.deck.infinite_random_card())
			self.states += [blackjack_state(player_hand.count(), d1, player_hand.get_usable_ace())]

			if dealer_hand.count() < 17:
				dealer_hand.add(self.deck.infinite_random_card())

		# ... and dealer always hits if score < 17:
		while dealer_hand.count() < 17:
			dealer_hand.add(self.deck.infinite_random_card())

		if player_hand.count() > 21:					# Lose game when bust
			return self.reward['lose'], self.states

		elif dealer_hand.count() > 21:					# Win if dealer busts
			return self.reward['win'], self.states

		elif player_hand.count() < dealer_hand.count():	# Lose game when score less than dealer
			return self.reward['lose'], self.states

		elif player_hand.count() > dealer_hand.count():	# Win game when score greatest
			return self.reward['win'], self.states

		else:											# Draw on even score
			return self.reward['draw'], self.states

		return self.reward['lose'], self.states


# Use first-visit Monte-Carlo to predict state-value scores
def first_visit_mc_prediction(deck, n_plays, discount_factor=1.0):

	# Initialise mc
	n_states = 200
	N = np.zeros((30,11,2))
	S = np.zeros((30,11,2))
	S += 0

	n = 0
	# Loop forever (for each episode)
	while (n < n_plays):
		# Generate an episode following the policy
		game = blackjack_game(deck)
		reward, states = game.play()

		for t, state in enumerate(states):
			N[state.current_sum][state.dealer_show][state.usable_ace] += 1
			S[state.current_sum][state.dealer_show][state.usable_ace] += reward
		
		n += 1

	V = np.divide(S, N, out=np.zeros_like(N), where=S!=0)
	
	return V

def plot_state_value(V):

	fig = plt.figure()
	ax = fig.gca(projection='3d')

	x_range = np.arange(12, 22)
	y_range = np.arange(1, 11)
	X, Y = np.meshgrid(x_range, y_range)
	Z = np.array([V[x,y,1] for x,y in zip(np.ravel(X), np.ravel(Y))]).reshape(X.shape)

	ax.set_xlabel('Player sum')
	ax.set_ylabel('Dealer showing')
	ax.set_zlabel('State-value')
	ax.set_title('Usable ace')

	# Hide grid lines
	ax.grid(False)

	ax.view_init(ax.elev, -120)


	# Plot the surface.
	surf = ax.plot_wireframe(X, Y, Z, color = "grey",
	                       linewidth=0.5, antialiased=False)

	# Customize the axes.
	ax.set_zlim(-1, 1)
	ax.zaxis.set_major_locator(LinearLocator(3))
	ax.set_xlim(12, 21)
	ax.xaxis.set_major_locator(LinearLocator(10))

	ylabels = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10']
	ax.set_yticklabels(ylabels)
	ax.yaxis.set_major_locator(LinearLocator(len(ylabels)))
	ax.set_ylim(ax.get_ylim()[::-1])
	#fig.gca().invert_yaxis()
	#ax.set_ylim(1, 10)
	


	plt.show()

def main():
	n_plays = 50000
	deck = standard_52_deck()
	V = first_visit_mc_prediction(deck, n_plays)
	plot_state_value(V)


if __name__ == '__main__':
	main()